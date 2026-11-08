"""Integration tests against a real PostgreSQL + TimescaleDB (set FIREDRONE_TEST_DSN)."""
import threading
from datetime import datetime, timezone

import pytest

from firedrone import db
from firedrone.config import ModelConfig, Settings
from firedrone.imaging import encode_jpeg
from firedrone.model import UnavailableModel, load_model
from firedrone.synthetic import SYNTHETIC_DEVICE, seed, synthetic_ir_frame
from firedrone.worker import Worker


def settings(**model_kw):
    m = ModelConfig(name="TEST_ONLY", version="test", training_domain="ir")
    for k, v in model_kw.items():
        setattr(m, k, v)
    return Settings(db_url=None, model=m, worker_batch_size=50)


def test_schema_is_hypertables_and_idempotent(conn):
    db.apply_schema(conn)  # second application must not fail
    hts = {r["hypertable_name"] for r in conn.execute("SELECT hypertable_name FROM timescaledb_information.hypertables")}
    assert {"frames", "fire_assessments"} <= hts
    assert conn.execute("SELECT count(*) AS n FROM timescaledb_information.continuous_aggregates").fetchone()["n"] == 1


def test_no_model_path_marks_unavailable(conn):
    seed(conn, 6)
    w = Worker(settings(), conn, backend=UnavailableModel("no model configured"))
    assert w.run_once() == 6
    rows = conn.execute("SELECT status, fire_present, severity_class, is_synthetic FROM fire_assessments").fetchall()
    assert len(rows) == 6
    assert all(r["status"] == "model_unavailable" and r["fire_present"] is None and r["severity_class"] is None
               and r["is_synthetic"] for r in rows)
    assert conn.execute("SELECT count(*) n FROM frames WHERE status='done'").fetchone()["n"] == 6


def test_onnx_end_to_end_and_tracking(conn, tmp_path):
    from make_test_onnx import build
    path = tmp_path / "TEST_ONLY.onnx"
    build(str(path))
    s = settings(backend="onnx", path=str(path))
    s.model.input.width = s.model.input.height = 32
    s.model.thresholds.min_sharpness = 0.0
    backend, sha = load_model(s.model)
    assert not isinstance(backend, UnavailableModel)
    seed(conn, 12)
    assert Worker(s, conn, backend, sha).run_once() == 12
    rows = conn.execute("SELECT status, fire_present, fire_probability, severity_class, validated, model_sha256 "
                        "FROM fire_assessments ORDER BY captured_at").fetchall()
    assert len(rows) == 12 and all(r["model_sha256"] == sha and r["validated"] is False for r in rows)
    assert all(r["fire_probability"] is not None for r in rows)
    assert {r["status"] for r in rows} <= {"ok", "uncertain"}
    conn.commit()
    conn.autocommit = True  # refresh_continuous_aggregate cannot run inside a transaction
    conn.execute("CALL refresh_continuous_aggregate('fire_severity_1min', NULL, NULL)")
    conn.autocommit = False
    agg = conn.execute("SELECT sum(n_frames) n, bool_and(any_synthetic) syn FROM fire_severity_1min").fetchone()
    assert agg["n"] == 12 and agg["syn"] is True
    latest = conn.execute("SELECT device_id FROM fire_latest_by_device").fetchall()
    assert [r["device_id"] for r in latest] == [SYNTHETIC_DEVICE]


def test_concurrent_workers_never_double_process(conn):
    seed(conn, 40)
    from firedrone import db as dbm
    from conftest import TEST_DSN
    counts = []

    def run():
        c = dbm.connect(TEST_DSN)
        w = Worker(settings(), c, backend=UnavailableModel("x"))
        w.s.worker_batch_size = 5
        n = 0
        while (k := w.run_once()):
            n += k
        counts.append(n)
        c.close()

    ts = [threading.Thread(target=run) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(counts) == 40
    assert conn.execute("SELECT count(*) n FROM fire_assessments").fetchone()["n"] == 40


def test_duplicate_frame_is_deduped(conn):
    db.upsert_device(conn, "d1")
    img = encode_jpeg(synthetic_ir_frame(3))
    ts = datetime.now(timezone.utc)
    assert db.insert_frame(conn, device_id="d1", captured_at=ts, image=img, width=160, height=120)
    assert db.insert_frame(conn, device_id="d1", captured_at=ts, image=img, width=160, height=120) is None


def test_inference_error_retries_then_records_error(conn):
    seed(conn, 1)

    class Boom:
        def run(self, x):
            raise RuntimeError("boom")

    s = settings(backend="onnx")
    s.worker_max_attempts = 2
    w = Worker(s, conn, backend=Boom(), model_sha=None)
    w.run_once()
    assert conn.execute("SELECT status FROM frames").fetchone()["status"] == "pending"
    w.run_once()
    r = conn.execute("SELECT status, fire_present FROM fire_assessments").fetchone()
    assert r["status"] == "error" and r["fire_present"] is None


def test_ingestor_with_fake_camera_and_spool(conn, tmp_path):
    from firedrone.ingest import Ingestor
    from firedrone.imu import ImuSample

    class Cam:
        def __init__(self):
            self.i = 0

        def grab_jpeg(self, q):
            self.i += 1
            return encode_jpeg(synthetic_ir_frame(self.i, seed=self.i)), 160, 120

    class Imu:
        def read(self):
            return ImuSample(1, 2, 3, 0, 0, 1)

    from conftest import TEST_DSN
    s = Settings(db_url=TEST_DSN, device_id="SYNTHETIC-pi", spool_dir=str(tmp_path / "spool"), capture_interval_s=0)
    up = {"ok": False}

    def factory():
        if not up["ok"]:
            raise ConnectionError("no link")
        return db.connect(TEST_DSN)

    ing = Ingestor(s, camera=Cam(), imu=Imu(), conn_factory=factory)
    ing.run(3)  # offline -> spooled
    assert len(list((tmp_path / "spool").glob("*.jpg"))) == 3
    up["ok"] = True
    ing.run(2)  # online -> flush 3 + insert 2
    assert not list((tmp_path / "spool").glob("*.jpg"))
    r = conn.execute("SELECT count(*) n, bool_and(imu_ok) imu FROM frames WHERE device_id='SYNTHETIC-pi'").fetchone()
    assert r["n"] == 5 and r["imu"] is True
