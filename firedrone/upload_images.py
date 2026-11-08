"""Upload REAL image files from a laptop folder into the frames table (no Pi, no camera, no IMU).

Rows get capture_source='file_replay', is_synthetic=FALSE, imu_ok=FALSE and the sensor domain you declare, so they
are never confused with live camera frames or generated test data. The inference worker processes them like any
other pending frame.

    python -m firedrone.upload_images demo_images --domain rgb --mission-id demo-1
    python -m firedrone.upload_images demo_images --domain rgb --interval 2      # simulate a 1 frame / 2 s feed

Re-running on the same folder skips images already stored for that device (same JPEG bytes), unless
--allow-duplicates is given.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import db
from .imaging import decode, encode_jpeg

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(folder: Path, recursive: bool) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def prepare(path: Path, max_side: int, quality: int):
    """Decode any supported format, optionally downscale, re-encode as JPEG (what the worker expects)."""
    img = decode(path.read_bytes())
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    data = encode_jpeg(img, quality)
    return data, img.width, img.height


def already_stored(conn, device_id: str, sha: str) -> bool:
    return conn.execute("SELECT 1 FROM frames WHERE device_id = %s AND sha256 = %s LIMIT 1",
                        (device_id, sha)).fetchone() is not None


def upload(conn, folder: Path, *, domain: str, device_id: str, mission_id: str | None, interval: float = 0.0,
           timestamps: str = "now", recursive: bool = False, allow_duplicates: bool = False, max_side: int = 1280,
           quality: int = 90, log=print) -> dict:
    files = find_images(folder, recursive)
    if not files:
        raise SystemExit(f"no images ({', '.join(sorted(IMAGE_EXTENSIONS))}) found in {folder}")
    db.upsert_device(conn, device_id, domain, camera_model="file_replay",
                     description="laptop folder replay (no Pi / no live camera)")
    stats = {"uploaded": 0, "skipped_duplicate": 0, "failed": 0, "frame_ids": []}
    for i, f in enumerate(files):
        t0 = time.monotonic()
        try:
            data, w, h = prepare(f, max_side, quality)
        except Exception as e:
            stats["failed"] += 1
            log(f"FAIL  {f.name}: cannot read image ({e})")
            continue
        sha = hashlib.sha256(data).hexdigest()
        if not allow_duplicates and already_stored(conn, device_id, sha):
            stats["skipped_duplicate"] += 1
            log(f"skip  {f.name} (already uploaded for {device_id})")
            continue
        ts = (datetime.fromtimestamp(f.stat().st_mtime, timezone.utc) if timestamps == "mtime"
              else datetime.now(timezone.utc))
        fid = db.insert_frame(conn, device_id=device_id, mission_id=mission_id, captured_at=ts, image=data,
                              width=w, height=h, sensor_domain=domain, imu=None, is_synthetic=False,
                              capture_source="file_replay", source_name=str(f.relative_to(folder)))
        if fid is None:
            stats["skipped_duplicate"] += 1
            log(f"skip  {f.name} (identical image at identical timestamp)")
            continue
        stats["uploaded"] += 1
        stats["frame_ids"].append(fid)
        log(f"ok    {f.name} -> frame {fid} ({w}x{h}, {len(data) // 1024} KB, {ts.isoformat(timespec='seconds')})")
        if interval > 0 and i < len(files) - 1:
            time.sleep(max(0.0, interval - (time.monotonic() - t0)))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--domain", required=True, choices=["rgb", "ir", "unknown"],
                    help="what these images really are (ordinary photos = rgb)")
    ap.add_argument("--device-id", default=os.environ.get("FIREDRONE_DEVICE_ID", "laptop-replay"))
    ap.add_argument("--mission-id", default=os.environ.get("FIREDRONE_MISSION_ID"))
    ap.add_argument("--interval", type=float, default=0.0, help="seconds between uploads (simulated live feed)")
    ap.add_argument("--timestamps", choices=["now", "mtime"], default="now",
                    help="captured_at = upload time (default) or the file's modification time")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--allow-duplicates", action="store_true")
    ap.add_argument("--max-side", type=int, default=1280, help="downscale longer side to this (0 = keep)")
    ap.add_argument("--init-schema", action="store_true", help="create/upgrade tables first (idempotent)")
    a = ap.parse_args(argv)
    if not a.folder.is_dir():
        sys.exit(f"not a folder: {a.folder}")
    conn = db.connect(os.environ.get("FIREDRONE_DB_URL"))
    if a.init_schema:
        db.apply_schema(conn)
    s = upload(conn, a.folder, domain=a.domain, device_id=a.device_id, mission_id=a.mission_id, interval=a.interval,
               timestamps=a.timestamps, recursive=a.recursive, allow_duplicates=a.allow_duplicates,
               max_side=a.max_side)
    print(f"done: {s['uploaded']} uploaded, {s['skipped_duplicate']} skipped (duplicates), {s['failed']} failed")


if __name__ == "__main__":
    main()
