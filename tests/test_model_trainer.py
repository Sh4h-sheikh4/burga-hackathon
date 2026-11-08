"""Code-validation tests for model/train_predict.py on SYNTHETIC blob images. They prove the code trains, exports,
calibrates and plugs into the pipeline - NOT that it detects real fires."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
import train_predict as tp  # noqa: E402

from firedrone.synthetic import synthetic_ir_frame  # noqa: E402

RADII = {"no_fire": 0, "low": 8, "moderate": 20, "high": 40}


def make_data(root: Path, per=(12, 5, 5), classes=RADII):
    for split, n, off in zip(("train", "val", "test"), per, (0, 1000, 2000)):
        for c, r in classes.items():
            d = root / split / c
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                synthetic_ir_frame(r, seed=off + i * 7 + r).save(d / f"{i}.png")
    return root


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sev")
    data = make_data(tmp / "data")
    tp.main(["train", "--data", str(data), "--out", str(tmp / "art"), "--domain", "ir", "--size", "32",
             "--epochs", "12", "--min-per-class", "10", "--batch-size", "8"])
    return tmp


def test_artifacts_and_config(trained):
    art = trained / "art"
    for f in ("fire_model.onnx", "fire_model.pt", "model_config.json", "metrics.json"):
        assert (art / f).exists()
    cfg = json.loads((art / "model_config.json").read_text())["model"]
    assert cfg["validated"] is False and cfg["backend"] == "onnx" and cfg["training_domain"] == "ir"
    assert cfg["output"]["labels"] == ["no_fire", "low", "moderate", "high"]
    assert cfg["input"]["layout"] == "NCHW" and cfg["input"]["width"] == 32
    m = json.loads((art / "metrics.json").read_text())
    assert "WARNING" in m and isinstance(m["test"], dict) and m["test"]["n"] == 20


def test_onnx_matches_torch_and_is_probability(trained):
    from firedrone.config import load_settings
    from firedrone.imaging import preprocess
    from firedrone.model import load_model
    art = trained / "art"
    s = load_settings(art / "model_config.json")
    backend, sha = load_model(s.model)
    x = preprocess(synthetic_ir_frame(20, seed=5), s.model.input)
    p_onnx = backend.run(x)[0]
    ck = torch.load(art / "fire_model.pt", weights_only=False)
    net = tp.build_model(4)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    with torch.no_grad():
        p_torch = (net(torch.from_numpy(x)) / ck["temperature"]).softmax(1).numpy()
    assert np.allclose(p_onnx, p_torch, atol=1e-4)
    assert abs(float(p_onnx.sum()) - 1.0) < 1e-4 and sha


def test_predict_cli_uses_pipeline_decision(trained, capsys):
    img = trained / "data" / "test" / "high" / "0.png"
    tp.main(["predict", str(img), "--config", str(trained / "art" / "model_config.json")])
    out = json.loads(capsys.readouterr().out)
    assert out["validated"] is False and out["status"] in ("ok", "uncertain")


def test_refuses_fire_only_dataset(tmp_path):
    data = make_data(tmp_path / "d", classes={"low": 8, "high": 40})
    with pytest.raises(SystemExit, match="no_fire"):
        tp.main(["train", "--data", str(data), "--classes", "low,high", "--out", str(tmp_path / "a"),
                 "--min-per-class", "1", "--epochs", "1", "--size", "32"])


def test_refuses_too_few_images(tmp_path):
    data = make_data(tmp_path / "d", per=(3, 2, 2))
    with pytest.raises(SystemExit, match="min-per-class"):
        tp.main(["train", "--data", str(data), "--out", str(tmp_path / "a"), "--epochs", "1", "--size", "32"])


def test_original_user_script_behaviour(tmp_path):
    """Runs the user's ORIGINAL train_predict.py unchanged: binary only, no probabilities, fails on fire-only data."""
    pytest.importorskip("sklearn")
    pytest.importorskip("cv2")
    orig = ROOT / "originals" / "user_upload" / "train_predict.py"
    if not orig.exists():
        pytest.skip("original not present")
    script = tmp_path / "orig.py"
    shutil.copy(orig, script)
    make_data(tmp_path / "data", per=(10, 4, 0), classes={"fire": 30, "no_fire": 0})
    r = subprocess.run([sys.executable, str(script), "train", "--data", str(tmp_path / "data"),
                        "--model", str(tmp_path / "m.joblib")], capture_output=True, text=True)
    assert r.returncode == 0 and "Saved model" in r.stdout
    r = subprocess.run([sys.executable, str(script), "predict", str(tmp_path / "data/val/fire/0.png"),
                        "--model", str(tmp_path / "m.joblib")], capture_output=True, text=True)
    assert r.stdout.strip() in ("Prediction: fire", "Prediction: no_fire")   # a label only, no severity/confidence
    import joblib
    assert not hasattr(joblib.load(tmp_path / "m.joblib"), "predict_proba")
    shutil.rmtree(tmp_path / "data" / "train" / "no_fire")
    r = subprocess.run([sys.executable, str(script), "train", "--data", str(tmp_path / "data"),
                        "--model", str(tmp_path / "m2.joblib")], capture_output=True, text=True)
    assert r.returncode != 0 and "no_fire" in (r.stdout + r.stderr)


def test_worker_with_trained_model(trained, conn):
    from firedrone.config import load_settings
    from firedrone.synthetic import seed
    from firedrone.worker import Worker
    s = load_settings(trained / "art" / "model_config.json")
    seed(conn, 9)
    w = Worker(s, conn)
    while w.run_once():
        pass
    rows = conn.execute("SELECT status, model_name, validated, class_probabilities FROM fire_assessments").fetchall()
    assert len(rows) == 9 and all(r["model_name"] == "fire_cnn_small_cnn" and r["validated"] is False for r in rows)
    assert all(set(r["class_probabilities"]) == {"no_fire", "low", "moderate", "high"} for r in rows)


def test_trainer_config_is_portable(trained, tmp_path):
    """model_config.json stores the ONNX path relative to itself, so the artifacts folder can be moved."""
    from firedrone.config import load_settings
    from firedrone.model import UnavailableModel, load_model
    cfg = json.loads((trained / "art" / "model_config.json").read_text())
    assert cfg["model"]["path"] == "fire_model.onnx"
    moved = tmp_path / "moved"
    shutil.copytree(trained / "art", moved)
    s = load_settings(moved / "model_config.json")
    backend, _ = load_model(s.model)
    assert not isinstance(backend, UnavailableModel)
