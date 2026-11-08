"""Pi-side capture: USB IR camera (UVC via OpenCV/V4L2) + LSM6DS3TR-C (SPI by default) -> frames table.

If the database is unreachable (drone out of range), frames are spooled to disk (JPEG + JSON sidecar)
and uploaded later by `--flush-spool` or automatically on the next successful connection.
    FIREDRONE_DB_URL=postgres://... python -m firedrone.ingest --config config/example.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import db
from .config import Settings, load_settings
from .imu import ImuSample

log = logging.getLogger("firedrone.ingest")


class Camera:
    def __init__(self, index: int, width: int, height: int):
        import cv2  # pip install opencv-python-headless (Pi: apt install python3-opencv)
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera /dev/video{index}")

    def grab_jpeg(self, quality: int) -> tuple[bytes, int, int]:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        ok, buf = self.cv2.imencode(".jpg", frame, [self.cv2.IMWRITE_JPEG_QUALITY, quality])
        h, w = frame.shape[:2]
        return buf.tobytes(), w, h


class Spool:
    def __init__(self, path: str):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)

    def put(self, meta: dict, image: bytes) -> None:
        # Name must be unique and must not contain '.', which Path.with_suffix would treat as an extension.
        safe_ts = re.sub(r"[^0-9A-Za-z]", "-", meta["captured_at"])
        base = f"{safe_ts}_{meta['device_id']}_{uuid.uuid4().hex[:8]}"
        (self.dir / f"{base}.jpg").write_bytes(image)
        tmp = self.dir / f"{base}.json.tmp"
        tmp.write_text(json.dumps(meta))
        tmp.rename(self.dir / f"{base}.json")  # JSON written last + atomically: a crash never leaves a half record

    def flush(self, conn) -> int:
        n = 0
        for js in sorted(self.dir.glob("*.json")):
            meta = json.loads(js.read_text())
            jpg = js.with_name(js.name[:-len(".json")] + ".jpg")
            imu = ImuSample(**meta["imu"]) if meta.get("imu") else None
            db.insert_frame(conn, device_id=meta["device_id"], mission_id=meta.get("mission_id"),
                            captured_at=datetime.fromisoformat(meta["captured_at"]), image=jpg.read_bytes(),
                            width=meta.get("width"), height=meta.get("height"),
                            sensor_domain=meta.get("sensor_domain", "ir"), imu=imu)
            jpg.unlink()
            js.unlink()
            n += 1
        return n


class Ingestor:
    def __init__(self, s: Settings, camera=None, imu=None, conn_factory=None):
        self.s = s
        self.camera = camera or Camera(s.camera_index, s.camera_width, s.camera_height)
        self.imu = imu
        if imu is None and s.imu_enabled:
            try:
                from .imu import open_imu
                self.imu = open_imu(s.imu_interface, s.imu_i2c_bus, s.imu_address, s.imu_spi_bus, s.imu_spi_device)
            except Exception as e:  # keep capturing images even if the IMU is missing
                log.warning("IMU unavailable (%s); frames stored with imu_ok=false", e)
        self.spool = Spool(s.spool_dir)
        self.conn_factory = conn_factory or (lambda: db.connect(s.db_url, connect_timeout=5))
        self.conn = None

    def _conn(self):
        if self.conn is None or self.conn.closed:
            self.conn = self.conn_factory()
            db.upsert_device(self.conn, self.s.device_id, self.s.sensor_domain)
            flushed = self.spool.flush(self.conn)
            if flushed:
                log.info("uploaded %d spooled frames", flushed)
        return self.conn

    def capture_one(self):
        image, w, h = self.camera.grab_jpeg(self.s.jpeg_quality)
        sample = None
        if self.imu is not None:
            try:
                sample = self.imu.read()
            except Exception as e:
                log.warning("IMU read failed: %s", e)
        ts = datetime.now(timezone.utc)
        try:
            return db.insert_frame(self._conn(), device_id=self.s.device_id, mission_id=self.s.mission_id,
                                   captured_at=ts, image=image, width=w, height=h,
                                   sensor_domain=self.s.sensor_domain, imu=sample)
        except Exception as e:
            log.warning("DB unavailable (%s); spooling frame locally", e)
            self.conn = None
            self.spool.put({"device_id": self.s.device_id, "mission_id": self.s.mission_id,
                            "captured_at": ts.isoformat(), "width": w, "height": h,
                            "sensor_domain": self.s.sensor_domain,
                            "imu": sample.__dict__ if sample else None}, image)
            return None

    def run(self, max_frames: int | None = None):
        n = 0
        while max_frames is None or n < max_frames:
            t0 = time.monotonic()
            self.capture_one()
            n += 1
            time.sleep(max(0.0, self.s.capture_interval_s - (time.monotonic() - t0)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--frames", type=int, help="stop after N frames")
    ap.add_argument("--flush-spool", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = load_settings(args.config)
    if args.flush_spool:
        print(Spool(s.spool_dir).flush(db.connect(s.db_url)), "frames uploaded")
        return
    Ingestor(s).run(args.frames)


if __name__ == "__main__":
    main()
