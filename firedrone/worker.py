"""Inference worker: pulls pending frames from the database, assesses them, writes fire_assessments back.

Run on a laptop/server (recommended) or on the Pi 3 with a TFLite/ONNX model.
    FIREDRONE_DB_URL=postgres://... python -m firedrone.worker --config config/example.json
"""
from __future__ import annotations

import argparse
import logging
import math
import socket
import time

from . import db
from .assessment import Assessment, decide
from .config import Settings, config_sha256, load_settings
from .imaging import decode, frame_stats, preprocess, quality_flags
from .model import UnavailableModel, interpret, load_model

log = logging.getLogger("firedrone.worker")


class Worker:
    def __init__(self, settings: Settings, conn, backend=None, model_sha=None):
        self.s = settings
        self.conn = conn
        if backend is None:
            backend, model_sha = load_model(settings.model)
        self.backend = backend
        self.model_sha = model_sha
        self.cfg_sha = config_sha256(settings.model)
        self.worker_id = f"{socket.gethostname()}:{id(self):x}"
        if isinstance(backend, UnavailableModel):
            log.warning("MODEL UNAVAILABLE: %s -> every frame will be stored as status=model_unavailable",
                        backend.reason)

    def assess_frame(self, frame: dict) -> Assessment:
        m = self.s.model
        img = decode(bytes(frame["image"]))
        stats = frame_stats(img, m.thresholds)
        gyro = None
        if frame.get("gyro_x") is not None:
            gyro = math.sqrt(frame["gyro_x"] ** 2 + frame["gyro_y"] ** 2 + frame["gyro_z"] ** 2)
        flags = quality_flags(stats, m.thresholds, gyro)
        if isinstance(self.backend, UnavailableModel):
            return decide(None, m, stats, flags, frame["sensor_domain"], self.backend.reason)
        outputs = self.backend.run(preprocess(img, m.input))
        return decide(interpret(outputs, m.output), m, stats, flags, frame["sensor_domain"])

    def run_once(self) -> int:
        frames = db.claim_frames(self.conn, self.worker_id, self.s.worker_batch_size,
                                 self.s.worker_stale_claim_s, self.s.worker_max_attempts)
        for fr in frames:
            try:
                a = self.assess_frame(fr)
            except Exception as e:
                log.exception("frame %s failed", fr["frame_id"])
                if fr["attempts"] >= self.s.worker_max_attempts:
                    a = Assessment(status="error", notes=[f"inference error: {e!r}"])
                else:
                    db.fail_frame(self.conn, fr, repr(e), self.s.worker_max_attempts)
                    continue
            db.write_assessment(self.conn, fr, a, model_name=self.s.model.name,
                                model_version=self.s.model.version, model_sha256=self.model_sha,
                                config_sha256=self.cfg_sha)
            log.info("frame %s -> status=%s fire=%s sev=%s p=%s", fr["frame_id"], a.status, a.fire_present,
                     a.severity_class, None if a.fire_probability is None else round(a.fire_probability, 3))
        return len(frames)

    def run_forever(self):
        while True:
            if self.run_once() == 0:
                time.sleep(self.s.worker_poll_s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--once", action="store_true", help="drain the pending queue once, then exit")
    ap.add_argument("--init-schema", action="store_true")
    ap.add_argument("--requeue", action="store_true",
                    help="re-process frames that are already done/failed (e.g. after training a new model)")
    ap.add_argument("--mission-id", help="limit --requeue to one mission")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = load_settings(args.config)
    conn = db.connect(s.db_url)
    if args.init_schema:
        db.apply_schema(conn)
    if args.requeue:
        log.info("requeued %d frames", db.requeue(conn, args.mission_id))
    w = Worker(s, conn)
    if args.once:
        n = 0
        while (k := w.run_once()):
            n += k
        log.info("processed %d frames", n)
    else:
        w.run_forever()


if __name__ == "__main__":
    main()
