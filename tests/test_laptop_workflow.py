"""Laptop-only workflow: real image files -> frames (file_replay) -> worker -> results. Uses SYNTHETIC images as the
'real files' stand-in; tests plumbing only."""
import json
import os
import time

import pytest

from firedrone import db
from firedrone.config import load_settings
from firedrone.synthetic import synthetic_ir_frame
from firedrone.upload_images import main as upload_main, upload


def make_folder(tmp_path, n=4):
    d = tmp_path / "demo_images"
    d.mkdir()
    for i in range(n):
        img = synthetic_ir_frame(10 * i, seed=i).convert("RGB")
        img.save(d / f"img_{i:02d}.png")
    (d / "notes.txt").write_text("not an image")          # ignored (extension)
    (d / "broken.jpg").write_bytes(b"not really a jpeg")   # counted as failed, does not abort
    return d


def test_relative_model_path_resolves_against_config(tmp_path):
    (tmp_path / "fire_model.onnx").write_bytes(b"x")
    cfg = tmp_path / "model_config.json"
    cfg.write_text(json.dumps({"model": {"backend": "onnx", "path": "fire_model.onnx"}}))
    s = load_settings(cfg)
    assert s.model.path == str((tmp_path / "fire_model.onnx").resolve())


def test_upload_folder_marks_file_replay_and_dedupes(conn, tmp_path):
    d = make_folder(tmp_path)
    s1 = upload(conn, d, domain="rgb", device_id="laptop-replay", mission_id="demo-1", log=lambda *_: None)
    assert (s1["uploaded"], s1["skipped_duplicate"], s1["failed"]) == (4, 0, 1)
    rows = conn.execute("SELECT capture_source, is_synthetic, imu_ok, sensor_domain, source_name, status, "
                        "image_format FROM frames ORDER BY source_name").fetchall()
    assert len(rows) == 4
    assert all(r["capture_source"] == "file_replay" and r["is_synthetic"] is False and r["imu_ok"] is False
               and r["sensor_domain"] == "rgb" and r["status"] == "pending" and r["image_format"] == "jpeg"
               for r in rows)
    assert [r["source_name"] for r in rows] == [f"img_{i:02d}.png" for i in range(4)]
    s2 = upload(conn, d, domain="rgb", device_id="laptop-replay", mission_id="demo-1", log=lambda *_: None)
    assert s2["uploaded"] == 0 and s2["skipped_duplicate"] == 4
    s3 = upload(conn, d, domain="rgb", device_id="laptop-replay", mission_id="demo-1", allow_duplicates=True,
                log=lambda *_: None)
    assert s3["uploaded"] == 4
    dev = conn.execute("SELECT sensor_domain, camera_model FROM devices WHERE device_id='laptop-replay'").fetchone()
    assert dev["sensor_domain"] == "rgb" and dev["camera_model"] == "file_replay"


def test_upload_mtime_and_pacing(conn, tmp_path):
    d = make_folder(tmp_path, n=2)
    os.utime(d / "img_00.png", (1_700_000_000, 1_700_000_000))
    t0 = time.monotonic()
    upload(conn, d, domain="unknown", device_id="dev-m", mission_id=None, timestamps="mtime", interval=0.3,
           log=lambda *_: None)
    assert time.monotonic() - t0 >= 0.25
    first = conn.execute("SELECT extract(epoch FROM captured_at)::bigint e FROM frames WHERE source_name='img_00.png'"
                         ).fetchone()["e"]
    assert first == 1_700_000_000


def test_cli_end_to_end_with_trained_style_config(conn, tmp_path, monkeypatch, capsys):
    """CLI upload -> worker (no model) -> results CLI; checks model_unavailable semantics and image retrieval."""
    from conftest import TEST_DSN
    from firedrone import results
    from firedrone.worker import main as worker_main
    monkeypatch.setenv("FIREDRONE_DB_URL", TEST_DSN)
    d = make_folder(tmp_path, n=3)
    upload_main([str(d), "--domain", "rgb", "--mission-id", "demo-cli"])
    cfg = tmp_path / "nomodel.json"
    cfg.write_text(json.dumps({"model": {"name": "none", "backend": "none"}}))
    worker_main(["--config", str(cfg), "--once"])
    r = conn.execute("SELECT status, fire_present FROM fire_assessments").fetchall()
    assert len(r) == 3 and all(x["status"] == "model_unavailable" and x["fire_present"] is None for x in r)
    capsys.readouterr()
    results.main(["--mission-id", "demo-cli"])
    out = capsys.readouterr().out
    assert "file_replay" in out and "model_unavailable" in out and "img_00.png" in out
    fid = conn.execute("SELECT frame_id FROM frames LIMIT 1").fetchone()["frame_id"]
    results.main(["--save-image", str(fid), "--out", str(tmp_path / "back.jpg")])
    assert (tmp_path / "back.jpg").read_bytes()[:2] == b"\xff\xd8"


def test_requeue_lets_a_new_model_rescore_old_frames(conn, tmp_path):
    from firedrone.config import ModelConfig, Settings
    from firedrone.model import UnavailableModel
    from firedrone.worker import Worker
    d = make_folder(tmp_path, n=2)
    upload(conn, d, domain="rgb", device_id="laptop-replay", mission_id="m1", log=lambda *_: None)
    Worker(Settings(db_url=None, model=ModelConfig(name="none", version="0")), conn,
           backend=UnavailableModel("x")).run_once()
    assert db.requeue(conn, mission_id="m1") == 2
    Worker(Settings(db_url=None, model=ModelConfig(name="other", version="1")), conn,
           backend=UnavailableModel("y")).run_once()
    rows = conn.execute("SELECT model_name, count(*) n FROM fire_assessments GROUP BY 1 ORDER BY 1").fetchall()
    assert [(r["model_name"], r["n"]) for r in rows] == [("none", 2), ("other", 2)]   # old outputs kept
