#!/usr/bin/env python3
"""Write a tiny SYNTHETIC dataset (grey blobs, NOT fire photos) to check that training, upload, the worker and the
database are wired correctly before real images are ready. A model trained on it means nothing about real fires.

    python tools/make_smoke_dataset.py            # -> smoke_data/{train,val,test}/<class>/  and  demo_images/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from firedrone.synthetic import synthetic_ir_frame  # noqa: E402

RADII = {"no_fire": 0, "low": 8, "moderate": 20, "high": 40}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="smoke_data")
    ap.add_argument("--demo", default="demo_images")
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--binary", action="store_true", help="only no_fire/fire folders (presence-only model)")
    a = ap.parse_args()
    classes = {"no_fire": 0, "fire": 25} if a.binary else RADII
    for split, n, off in (("train", a.per_class, 0), ("val", max(5, a.per_class // 3), 10_000),
                          ("test", max(5, a.per_class // 3), 20_000)):
        for c, r in classes.items():
            d = Path(a.out) / split / c
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                synthetic_ir_frame(r, seed=off + i * 13 + r).convert("RGB").save(d / f"SYNTHETIC_{c}_{i:03d}.png")
    demo = Path(a.demo)
    demo.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate([0, 0, 6, 12, 20, 30, 40, 45]):
        synthetic_ir_frame(r, seed=50_000 + i).convert("RGB").save(demo / f"SYNTHETIC_demo_{i:02d}.png")
    print(f"wrote SYNTHETIC data to {a.out}/ and {a.demo}/ (plumbing test only)")


if __name__ == "__main__":
    main()
