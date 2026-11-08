"""SYNTHETIC test fixtures. These are generated grey-scale blobs, NOT fire imagery and NOT model results.
Every row inserted from here has is_synthetic = TRUE and mission_id starting with 'SYNTHETIC'.
    FIREDRONE_DB_URL=... python -m firedrone.synthetic --frames 20
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image

from . import db
from .imaging import encode_jpeg
from .imu import ImuSample

SYNTHETIC_DEVICE = "SYNTHETIC-drone"
SYNTHETIC_MISSION = "SYNTHETIC-test-mission"


def synthetic_ir_frame(hot_radius: int, width=160, height=120, seed=0) -> Image.Image:
    """Dark noisy background with an optional bright disc (hot_radius=0 -> none). Generic grayscale test pattern;
    it does not model any specific near-IR or thermal camera."""
    rng = np.random.default_rng(seed)
    img = rng.normal(40, 8, (height, width))
    # mild structure so the blur metric is not zero
    img += 10 * np.sin(np.linspace(0, 12, width))[None, :]
    if hot_radius > 0:
        yy, xx = np.mgrid[:height, :width]
        d = np.sqrt((yy - height / 2) ** 2 + (xx - width / 2) ** 2)
        img = np.where(d < hot_radius, 250, img)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")


def seed(conn, n_frames: int = 12, start: datetime | None = None) -> list:
    db.upsert_device(conn, SYNTHETIC_DEVICE, "ir", camera_model="synthetic", description="SYNTHETIC test data")
    start = start or datetime.now(timezone.utc) - timedelta(seconds=n_frames)
    ids = []
    for i in range(n_frames):
        radius = 0 if i < n_frames // 3 else int(5 + 40 * (i / n_frames))  # growing "hotspot"
        img = synthetic_ir_frame(radius, seed=i)
        ids.append(db.insert_frame(conn, device_id=SYNTHETIC_DEVICE, mission_id=SYNTHETIC_MISSION,
                                   captured_at=start + timedelta(seconds=i), image=encode_jpeg(img),
                                   width=img.width, height=img.height, sensor_domain="ir",
                                   imu=ImuSample(1.0, -0.5, 0.2, 0.0, 0.0, 1.0), is_synthetic=True))
    return ids


def main(argv=None):
    import os
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--init-schema", action="store_true")
    a = ap.parse_args(argv)
    conn = db.connect(os.environ.get("FIREDRONE_DB_URL"))
    if a.init_schema:
        db.apply_schema(conn)
    print(len([i for i in seed(conn, a.frames) if i]), "SYNTHETIC frames inserted")


if __name__ == "__main__":
    main()
