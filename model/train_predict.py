#!/usr/bin/env python3
"""Fire presence + severity CNN: train, evaluate, export (ONNX), predict.

Revised replacement for the user's original `train_predict.py` (kept unmodified in originals/user_upload/).
The original was a LinearSVC on raw grayscale pixels with labels fire/no_fire only: it could not output severity,
had no confidence scores, and cannot train on a fire-only dataset. This version:

  * is a small CNN (PyTorch) that predicts one of the classes you label, e.g. no_fire / low / moderate / high
    (presence = 1 - P(no_fire); severity = most likely fire class), or presence only with --classes no_fire,fire;
  * refuses to train without a no_fire class or with too few images per class;
  * uses exactly the same preprocessing code as the TigerData inference worker (firedrone.imaging.preprocess);
  * fits a softmax temperature on the validation split so probabilities are better calibrated for the
    worker's uncertainty thresholds, and bakes it into the exported ONNX model;
  * writes fire_model.onnx + model_config.json (drop-in "model" block for the pipeline) + metrics.json.

Nothing here makes accuracy claims: metrics.json reports what the model did on YOUR val/test folders, and
model_config.json always has "validated": false until you review the test-set metrics and set it yourself.

Data layout (class folder names must match --classes; images never shared between splits, ideally different
flights/fires per split):
    data/train/no_fire/*.jpg  data/train/low/*.jpg  data/train/moderate/*.jpg  data/train/high/*.jpg
    data/val/...              data/test/...

    python model/train_predict.py train --data data --out artifacts/ --domain ir --epochs 30
    python model/train_predict.py predict some_frame.jpg --config artifacts/model_config.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from firedrone.config import InputSpec, ModelConfig, OutputSpec, Thresholds  # noqa: E402
from firedrone.imaging import decode, preprocess  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASSES = "no_fire,low,moderate,high"
NO_FIRE = "no_fire"


def image_files(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def scan_split(root: Path, split: str, classes: list[str]) -> list[tuple[Path, int]]:
    items = []
    for idx, c in enumerate(classes):
        items += [(p, idx) for p in image_files(root / split / c)]
    return items


def check_dataset(train, classes, min_per_class):
    counts = {c: sum(1 for _, y in train if y == i) for i, c in enumerate(classes)}
    problems = []
    if NO_FIRE not in classes:
        problems.append(f"--classes must include '{NO_FIRE}': without negative examples the model cannot learn "
                        "whether a fire is present.")
    for c, n in counts.items():
        if n < min_per_class:
            problems.append(f"class '{c}' has {n} training images (< --min-per-class {min_per_class})")
    return counts, problems


# ----------------------------------------------------------------------------------------------------------- model
def build_model(n_classes: int, arch: str = "small_cnn", pretrained: bool = False):
    import torch.nn as nn
    if arch == "small_cnn":
        def block(i, o):
            # GroupNorm, not BatchNorm: BN running statistics are unreliable with a few hundred images / small batches
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.GroupNorm(8, o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.GroupNorm(8, o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        return nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))
    if arch == "mobilenet_v3_small":  # transfer learning; usually far better with only hundreds of images
        import torchvision  # pip install torchvision
        w = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None  # downloads weights
        m = torchvision.models.mobilenet_v3_small(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
        return m
    raise SystemExit(f"unknown --arch {arch}")


class _Probs:
    """Wraps logits model -> temperature-scaled softmax probabilities (what gets exported)."""

    @staticmethod
    def make(model, temperature: float):
        import torch.nn as nn

        class P(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model
                self.t = float(temperature)

            def forward(self, x):
                return (self.m(x) / self.t).softmax(dim=1)
        return P()


# ------------------------------------------------------------------------------------------------------------ data
def augment(img, rng: random.Random):
    from PIL import ImageEnhance, ImageOps
    img = img.convert("RGB")
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    img = img.rotate(rng.uniform(-10, 10), resample=2)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    w, h = img.size
    s = rng.uniform(0.85, 1.0)
    cw, ch = int(w * s), int(h * s)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    return img.crop((x0, y0, x0 + cw, y0 + ch))


def batches(items, spec: InputSpec, bs: int, train: bool, rng: random.Random):
    import torch
    order = list(range(len(items)))
    if train:
        rng.shuffle(order)
    for i in range(0, len(order), bs):
        xs, ys = [], []
        for j in order[i:i + bs]:
            p, y = items[j]
            img = decode(p.read_bytes())
            if train:
                img = augment(img, rng)
            xs.append(preprocess(img, spec)[0])
            ys.append(y)
        yield torch.from_numpy(np.stack(xs)), torch.tensor(ys)


def collect_logits(model, items, spec, bs=32):
    import torch
    model.eval()
    L, Y = [], []
    with torch.no_grad():
        for x, y in batches(items, spec, bs, False, random.Random(0)):
            L.append(model(x))
            Y.append(y)
    return torch.cat(L), torch.cat(Y)


def fit_temperature(logits, labels) -> float:
    import torch
    t = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=200)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / t.clamp(min=0.05), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.detach().clamp(min=0.05, max=20.0))


def metrics(probs: np.ndarray, y: np.ndarray, classes: list[str], th: Thresholds) -> dict:
    neg = classes.index(NO_FIRE)
    pred = probs.argmax(1)
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    per = {}
    for i, c in enumerate(classes):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        per[c] = {"support": int(cm[i].sum()),
                  "precision": None if tp + fp == 0 else round(float(tp / (tp + fp)), 4),
                  "recall": None if tp + fn == 0 else round(float(tp / (tp + fn)), 4)}
    p_fire = 1.0 - probs[:, neg]
    is_fire = y != neg
    fire_pred = p_fire >= th.fire_present_at_or_above
    nofire_pred = p_fire <= th.no_fire_at_or_below
    unc = ~(fire_pred | nofire_pred)
    out = {"n": int(len(y)), "classes": classes, "confusion_matrix_rows_truth_cols_pred": cm.tolist(),
           "accuracy_argmax": round(float((pred == y).mean()), 4) if len(y) else None, "per_class": per,
           "presence_at_thresholds": {
               "uncertain_rate": round(float(unc.mean()), 4) if len(y) else None,
               "fire_recall_confident": None if is_fire.sum() == 0 else round(float(fire_pred[is_fire].mean()), 4),
               "false_alarm_rate_confident": None if (~is_fire).sum() == 0 else
               round(float(fire_pred[~is_fire].mean()), 4)}}
    sev = [i for i in range(k) if i != neg]
    if len(sev) > 1:  # ordinal error between severity classes, on true-fire images predicted as fire
        m = is_fire & (pred != neg)
        if m.any():
            rank = {c: r for r, c in enumerate(sev)}
            out["severity_mean_abs_class_error"] = round(float(np.mean([abs(rank[t] - rank[p])
                                                                        for t, p in zip(y[m], pred[m])])), 4)
    return out


# ----------------------------------------------------------------------------------------------------------- train
def train(args):
    import torch
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    root = Path(args.data)
    tr, va, te = (scan_split(root, s, classes) for s in ("train", "val", "test"))
    counts, problems = check_dataset(tr, classes, args.min_per_class)
    print("train counts:", counts, "| val:", len(va), "| test:", len(te))
    if problems:
        raise SystemExit("Cannot train:\n  - " + "\n  - ".join(problems))
    if not va:
        raise SystemExit("A val/ split is required (early stopping + probability calibration).")
    spec = InputSpec(width=args.size, height=args.size, channels=3, layout="NCHW",
                     scale="imagenet" if args.arch == "mobilenet_v3_small" else "0_1")
    model = build_model(len(classes), args.arch, args.pretrained)
    w = torch.tensor([len(tr) / (len(classes) * max(counts[c], 1)) for c in classes], dtype=torch.float32)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_state, patience = float("inf"), None, 0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in batches(tr, spec, args.batch_size, True, rng):
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        vl, vy = collect_logits(model, va, spec)
        vloss = float(torch.nn.functional.cross_entropy(vl, vy))
        print(f"epoch {ep + 1}/{args.epochs} train_loss={tot / len(tr):.4f} val_loss={vloss:.4f}")
        if vloss < best - 1e-4:
            best, patience = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print("early stop")
                break
    model.load_state_dict(best_state)
    vl, vy = collect_logits(model, va, spec)
    T = fit_temperature(vl, vy)
    print(f"fitted softmax temperature T={T:.3f} on val")
    calib_warning = None
    if T >= 19.9 or T <= 0.06:
        calib_warning = (f"temperature hit its bound (T={T:.2f}): the model's val predictions are close to "
                         "uninformative - expect most outputs to be 'uncertain'. More/better data is needed.")
        print("WARNING:", calib_warning)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    probs_model = _Probs.make(model, T).eval()
    onnx_path = out / "fire_model.onnx"
    dummy = torch.zeros(1, 3, args.size, args.size)
    torch.onnx.export(probs_model, dummy, str(onnx_path), input_names=["x"], output_names=["probs"],
                      opset_version=17, dynamo=False)
    torch.save({"state_dict": model.state_dict(), "classes": classes, "arch": args.arch, "size": args.size,
                "temperature": T}, out / "fire_model.pt")
    th = Thresholds()
    binary = [c for c in classes if c != NO_FIRE]
    cfg = ModelConfig(name="fire_cnn_" + args.arch, version=args.version, backend="onnx",
                      path=onnx_path.name,  # relative to model_config.json (resolved by load_settings)
                      training_domain=args.domain, validated=False, input=spec,
                      output=OutputSpec(type="softmax_classes", labels=classes, no_fire_label=NO_FIRE,
                                        severity_labels=binary), thresholds=th)
    (out / "model_config.json").write_text(json.dumps({"model": asdict(cfg)}, indent=2))
    report = {"WARNING": "Metrics describe only the supplied val/test folders. They are not proof of real-world "
                         "accuracy; check that test images come from different flights/fires than train.",
              "train_counts": counts, "temperature": T, "calibration_warning": calib_warning, "arch": args.arch, "input": asdict(spec),
              "training_domain": args.domain,
              "val": metrics(_softmax_np(vl, T), vy.numpy(), classes, th)}
    if te:
        tl, ty = collect_logits(model, te, spec)
        report["test"] = metrics(_softmax_np(tl, T), ty.numpy(), classes, th)
    else:
        report["test"] = "no test/ split supplied - model is NOT evaluated on held-out data"
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {onnx_path}, {out / 'model_config.json'}, {out / 'metrics.json'}")
    print("model_config.json has validated=false. Review metrics.json (test split) before trusting outputs.")


def _softmax_np(logits, T):
    return (logits / T).softmax(dim=1).numpy()


# --------------------------------------------------------------------------------------------------------- predict
def predict(args):
    from firedrone.assessment import decide
    from firedrone.config import load_settings
    from firedrone.imaging import frame_stats, quality_flags
    from firedrone.model import UnavailableModel, interpret, load_model
    s = load_settings(args.config)
    backend, _ = load_model(s.model)
    if isinstance(backend, UnavailableModel):
        raise SystemExit(backend.reason)
    img = decode(Path(args.image).read_bytes())
    st = frame_stats(img, s.model.thresholds)
    a = decide(interpret(backend.run(preprocess(img, s.model.input)), s.model.output), s.model, st,
               quality_flags(st, s.model.thresholds), args.domain)
    print(json.dumps({k: v for k, v in a.__dict__.items()}, indent=2, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train or use a fire presence + severity CNN.")
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data")
    t.add_argument("--out", default="artifacts")
    t.add_argument("--classes", default=DEFAULT_CLASSES, help="comma list incl. no_fire; order = severity order")
    t.add_argument("--domain", choices=["rgb", "ir", "unknown"], default="unknown",
                   help="what the training images are; 'ir' should mean the same camera type as the drone's")
    t.add_argument("--arch", choices=["small_cnn", "mobilenet_v3_small"], default="small_cnn")
    t.add_argument("--pretrained", action="store_true", help="download ImageNet weights (mobilenet only)")
    t.add_argument("--size", type=int, default=128)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--patience", type=int, default=6)
    t.add_argument("--min-per-class", type=int, default=30)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--version", default="0.1.0")
    t.set_defaults(run=train)
    p = sub.add_parser("predict")
    p.add_argument("image")
    p.add_argument("--config", default="artifacts/model_config.json")
    p.add_argument("--domain", default="ir", help="sensor domain of this image")
    p.set_defaults(run=predict)
    args = ap.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
