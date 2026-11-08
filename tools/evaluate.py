#!/usr/bin/env python3
"""Evaluate the configured model on a LABELED HELD-OUT folder (images never used in training):

    heldout/no_fire/*.jpg  heldout/low/*.jpg  heldout/moderate/*.jpg  heldout/high/*.jpg

Prints a confusion matrix, per-class precision/recall, fire-detection recall/false-alarm rate and the
fraction of frames the thresholds leave 'uncertain'. Only set model.validated=true after this looks good
on IR images from YOUR camera.
    python tools/evaluate.py --config config/my_model.json --data heldout/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from firedrone.assessment import decide  # noqa: E402
from firedrone.config import load_settings  # noqa: E402
from firedrone.imaging import decode, frame_stats, preprocess, quality_flags  # noqa: E402
from firedrone.model import UnavailableModel, interpret, load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--domain", default="ir")
    a = ap.parse_args()
    s = load_settings(a.config)
    m = s.model
    backend, _ = load_model(m)
    if isinstance(backend, UnavailableModel):
        sys.exit(f"model unavailable: {backend.reason}")
    neg = m.output.no_fire_label
    conf, pairs = Counter(), []
    for cls_dir in sorted(p for p in Path(a.data).iterdir() if p.is_dir()):
        for f in sorted(cls_dir.glob("*")):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            img = decode(f.read_bytes())
            st = frame_stats(img, m.thresholds)
            out = interpret(backend.run(preprocess(img, m.input)), m.output)
            asm = decide(out, m, st, quality_flags(st, m.thresholds), a.domain)
            pred = "uncertain" if asm.status != "ok" else (neg if not asm.fire_present else
                                                         asm.severity_class or "fire_unknown_severity")
            conf[(cls_dir.name, pred)] += 1
            pairs.append((cls_dir.name, pred))
    if not pairs:
        sys.exit("no images found")
    truths = sorted({t for t, _ in pairs})
    preds = sorted({p for _, p in pairs})
    print("confusion (rows=truth, cols=pred):\n" + "truth\\pred".ljust(14) + "".join(p[:12].ljust(13) for p in preds))
    for t in truths:
        print(t[:13].ljust(14) + "".join(str(conf[(t, p)]).ljust(13) for p in preds))
    fire_t = [p for t, p in pairs if t != neg]
    neg_t = [p for t, p in pairs if t == neg]
    res = {
        "n": len(pairs),
        "uncertain_rate": sum(p == "uncertain" for _, p in pairs) / len(pairs),
        "fire_recall": (sum(p not in (neg, "uncertain") for p in fire_t) / len(fire_t)) if fire_t else None,
        "false_alarm_rate": (sum(p not in (neg, "uncertain") for p in neg_t) / len(neg_t)) if neg_t else None,
        "severity_exact_accuracy_on_confident_fire": None,
    }
    conf_fire = [(t, p) for t, p in pairs if t != neg and p not in (neg, "uncertain")]
    if conf_fire:
        res["severity_exact_accuracy_on_confident_fire"] = sum(t == p for t, p in conf_fire) / len(conf_fire)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
