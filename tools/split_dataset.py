"""Split two folders of images (fire / no fire) into the train/val layout train.py expects.

Example:
    python split_data.py --fire path/to/fire_images --no-fire path/to/no_fire_images --out data

Your original images are copied, not moved or changed.
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def split_class(src, out, name, val_frac, seed):
    files = sorted(p for p in Path(src).rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"No images found in {src}")
    random.Random(seed).shuffle(files)
    n_val = max(1, int(len(files) * val_frac))
    splits = {"val": files[:n_val], "train": files[n_val:]}
    for split, items in splits.items():
        dest = Path(out) / split / name
        dest.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(items):
            shutil.copy2(f, dest / f"{i:05d}_{f.name}")
    print(f"{name}: {len(splits['train'])} train, {len(splits['val'])} val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", required=True, help="folder of fire images")
    ap.add_argument("--no-fire", required=True, help="folder of non-fire images")
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    split_class(args.fire, args.out, "fire", args.val_frac, args.seed)
    split_class(args.no_fire, args.out, "no_fire", args.val_frac, args.seed)


if __name__ == "__main__":
    main()
