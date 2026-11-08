#!/usr/bin/env python3
"""Static review of a model training/definition .py file (no execution).

Answers: which framework, what the final layer emits, whether a fire/no-fire decision AND a severity
output exist, what loss is used, whether weights are saved/loaded, and common defects.
    python tools/inspect_model_source.py path/to/model.py [--json]
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


def _const(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node) if node is not None else None


def inspect(src: str) -> dict:
    r = {"framework": None, "syntax_ok": True, "layers": [], "final_layer": None, "losses": [],
         "class_count_hints": [], "saves_weights": False, "loads_weights": False, "data_dirs": [],
         "has_fire_presence_output": None, "has_severity_output": None, "estimators": [], "label_sets": [],
         "has_conv_layers": False, "findings": []}
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        r["syntax_ok"] = False
        r["findings"].append(f"SyntaxError line {e.lineno}: {e.msg}")
        return r
    imports = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
               and n.names} | {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    r["framework"] = "tensorflow/keras" if imports & {"tensorflow", "keras"} else "pytorch" if "torch" in imports else \
        "sklearn" if "sklearn" in imports else None
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn = ast.unparse(n.func)
        short = fn.split(".")[-1]
        kw = {k.arg: _const(k.value) for k in n.keywords if k.arg}
        if short in ("Dense", "Linear", "Conv2D", "Conv2d", "MaxPooling2D", "MaxPool2d", "Flatten", "Dropout",
                     "GlobalAveragePooling2D", "BatchNormalization", "BatchNorm2d"):
            args = [_const(a) for a in n.args]
            r["layers"].append({"type": short, "args": args, "kwargs": kw, "line": n.lineno})
        if short == "compile" and "loss" in kw:
            r["losses"].append(kw["loss"])
        if short in ("CrossEntropyLoss", "BCELoss", "BCEWithLogitsLoss", "MSELoss"):
            r["losses"].append(short)
        if short in ("LinearSVC", "SVC", "LogisticRegression", "RandomForestClassifier", "KNeighborsClassifier",
                     "MLPClassifier", "SGDClassifier", "GradientBoostingClassifier"):
            r["estimators"].append({"type": short, "kwargs": kw, "line": n.lineno})
        if fn in ("joblib.dump", "pickle.dump", "dump"):
            r["saves_weights"] = True
        if fn in ("joblib.load", "pickle.load"):
            r["loads_weights"] = True
        if short in ("save", "save_weights", "save_model") or fn.endswith("torch.save"):
            r["saves_weights"] = True
        if short in ("load_model", "load_weights", "load_state_dict") or fn.endswith("torch.load"):
            r["loads_weights"] = True
        if short in ("flow_from_directory", "image_dataset_from_directory", "ImageFolder"):
            r["data_dirs"].append({"call": short, "args": [_const(a) for a in n.args], "line": n.lineno})
        if short in ("flow_from_directory", "image_dataset_from_directory") and "class_mode" in kw:
            r["class_count_hints"].append(f"class_mode={kw['class_mode']}")
    for n in ast.walk(tree):
        if isinstance(n, (ast.Tuple, ast.List, ast.Set)) and n.elts and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str) for e in n.elts):
            vals = [e.value for e in n.elts]
            if any("fire" in v.lower() for v in vals):
                r["label_sets"].append({"labels": vals, "line": n.lineno})
    r["has_conv_layers"] = any(l["type"] in ("Conv2D", "Conv2d") for l in r["layers"])
    sev_words = re.compile(r"low|moderate|medium|high|severe|severity|mild|extreme|small|large", re.I)
    if r["label_sets"]:
        labels = {v for ls in r["label_sets"] for v in ls["labels"]}
        if not any(sev_words.search(v) for v in labels) and len(labels) <= 2:
            r["has_fire_presence_output"] = True
            r["has_severity_output"] = False
            r["findings"].append(f"Labels are only {sorted(labels)}: binary fire/no-fire classifier; it has no severity "
                                 "labels or output, so it cannot predict how bad a fire is.")
    for e in r["estimators"]:
        r["has_fire_presence_output"] = r["has_fire_presence_output"] or True
        if e["type"] in ("LinearSVC", "SGDClassifier"):
            r["findings"].append(f"{e['type']} (line {e['line']}) has no predict_proba: no calibrated confidence, so "
                                 "uncertain cases cannot be thresholded (use LogisticRegression/CalibratedClassifierCV or a CNN with softmax).")
        if not r["has_conv_layers"]:
            r["findings"].append(f"Model is a classical {e['type']} (line {e['line']}), not a CNN - no convolutional layers.")
    if re.search(r"reshape\(\s*-1\s*\)", src) and r["estimators"]:
        r["findings"].append("Features are raw flattened pixels (reshape(-1)): a linear model on raw pixels has no "
                             "translation/scale invariance and typically generalises poorly to new scenes.")
    if re.search(r"IMREAD_GRAYSCALE|convert\(['\"]L['\"]\)", src):
        r["findings"].append("Images are loaded as grayscale: colour is discarded (a loss if training data is RGB; may be acceptable for single-channel IR frames).")
    heads = [l for l in r["layers"] if l["type"] in ("Dense", "Linear")]
    if heads:
        last = heads[-1]
        units = last["args"][-1] if last["type"] == "Linear" and last["args"] else \
            (last["args"][0] if last["args"] else last["kwargs"].get("units"))
        act = last["kwargs"].get("activation")
        r["final_layer"] = {"type": last["type"], "units": units, "activation": act, "line": last["line"]}
        if isinstance(units, int):
            r["has_fire_presence_output"] = True
            r["has_severity_output"] = units >= 3  # heuristic only: regression/ordinal heads may differ; verify labels
            if units == 1:
                r["findings"].append("Final layer has 1 unit: likely binary fire/no-fire only (unless it is a regression "
                                     "head trained on a severity target); check the labels.")
            elif units == 2:
                r["findings"].append("Final layer has 2 units: likely two classes only; no severity unless trained with severity targets.")
            if units == 1 and act == "softmax":
                r["findings"].append("BUG: softmax over 1 unit always outputs 1.0 - use sigmoid.")
            if units >= 2 and act == "sigmoid" and any("categorical" in str(l) for l in r["losses"]):
                r["findings"].append("Mismatch: sigmoid multi-unit head with categorical loss; use softmax.")
    losses = " ".join(map(str, r["losses"]))
    if "binary_crossentropy" in losses and r["final_layer"] and r["final_layer"].get("units", 1) not in (1, None):
        r["findings"].append("binary_crossentropy with >1 output unit: check labels are multi-label not multi-class.")
    if not r["saves_weights"] and not r["loads_weights"]:
        r["findings"].append("No model save/load call found: trained weights may never be persisted.")
    if re.search(r"fire", src, re.I) and not re.search(r"no[_ ]?fire|non[_ ]?fire|nofire|negative", src, re.I):
        r["findings"].append("No negative ('no_fire') class referenced: a fire-only dataset cannot learn fire presence.")
    if not re.search(r"validation|val_|test_split|validation_split|train_test_split", src, re.I):
        r["findings"].append("No validation/test split found: accuracy cannot be measured.")
    if r["framework"] is None:
        r["findings"].append("No ML framework import found.")
    return r


def main():
    p = Path(sys.argv[1])
    r = inspect(p.read_text(errors="replace"))
    if "--json" in sys.argv:
        print(json.dumps(r, indent=2, default=str))
        return
    print(f"File: {p}\nFramework: {r['framework']}  syntax_ok={r['syntax_ok']}")
    print(f"Final layer: {r['final_layer']}\nLosses: {r['losses']}")
    print(f"Fire presence output: {r['has_fire_presence_output']}  Severity output: {r['has_severity_output']}")
    print("Findings:")
    for f in r["findings"]:
        print("  -", f)


if __name__ == "__main__":
    main()
