import json

import numpy as np
import pytest

from firedrone.assessment import decide
from firedrone.config import ModelConfig, OutputSpec, Thresholds, load_settings, validate_model_config
from firedrone.imaging import FrameStats
from firedrone.model import interpret, load_model, UnavailableModel

STATS = FrameStats(50, 200, 0.0, 0.01, 50)


def cfg(**kw):
    c = ModelConfig(name="t", version="1", backend="onnx", training_domain="ir", validated=False)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def softmax_out(p):
    return [np.array([p], dtype=np.float32)]


def test_no_model_gives_unknown_not_false():
    a = decide(None, cfg(), STATS, [], "ir", "no model")
    assert a.status == "model_unavailable" and a.fire_present is None and a.severity_class is None
    assert a.proxy_hot_pixel_fraction == 0.01  # proxy still recorded, separately


def test_confident_fire_with_severity():
    pred = interpret(softmax_out([0.05, 0.1, 0.75, 0.1]), OutputSpec())
    a = decide(pred, cfg(), STATS, [], "ir")
    assert a.status == "ok" and a.fire_present is True
    assert a.severity_class == "moderate" and a.severity_index == 1
    assert a.validated is False and any("NOT validated" in n for n in a.notes)


def test_confident_no_fire():
    a = decide(interpret(softmax_out([0.9, 0.05, 0.03, 0.02]), OutputSpec()), cfg(), STATS, [], "ir")
    assert a.status == "ok" and a.fire_present is False and a.severity_class is None


def test_grey_zone_is_uncertain():
    a = decide(interpret(softmax_out([0.5, 0.2, 0.2, 0.1]), OutputSpec()), cfg(), STATS, [], "ir")
    assert a.status == "uncertain" and a.fire_present is None


def test_low_severity_confidence_withheld():
    a = decide(interpret(softmax_out([0.0, 0.34, 0.33, 0.33]), OutputSpec()), cfg(), STATS, [], "ir")
    assert a.fire_present is True and a.severity_class is None and a.severity_confidence == pytest.approx(0.34)


def test_domain_mismatch_downgrades():
    a = decide(interpret(softmax_out([0.0, 0.0, 1.0, 0.0]), OutputSpec()), cfg(training_domain="rgb"), STATS, [], "ir")
    assert a.domain_mismatch and a.status == "uncertain" and a.fire_present is None


def test_low_quality_downgrades():
    a = decide(interpret(softmax_out([0.0, 0.0, 1.0, 0.0]), OutputSpec()), cfg(), STATS, ["too_dark"], "ir")
    assert a.status == "uncertain"


def test_binary_model_has_no_severity():
    spec = OutputSpec(type="binary_sigmoid")
    a = decide(interpret([np.array([[0.95]])], spec), cfg(output=spec), STATS, [], "ir")
    assert a.fire_present is True and a.severity_class is None
    assert any("no severity output" in n for n in a.notes)


def test_two_head_logits():
    spec = OutputSpec(type="two_head", severity_labels=["low", "high"], apply_softmax=True, apply_sigmoid=True)
    pred = interpret([np.array([[4.0]]), np.array([[0.0, 3.0]])], spec)
    assert pred.fire_probability > 0.98 and max(pred.severity_probs, key=pred.severity_probs.get) == "high"


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError):
        interpret(softmax_out([0.5, 0.5]), OutputSpec())


def test_threshold_validation():
    c = cfg(thresholds=Thresholds(fire_present_at_or_above=0.2, no_fire_at_or_below=0.8))
    with pytest.raises(ValueError):
        validate_model_config(c)


def test_settings_never_read_db_url_from_file(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"db_url": "postgres://leak", "device_id": "d9"}))
    monkeypatch.delenv("FIREDRONE_DB_URL", raising=False)
    s = load_settings(p)
    assert s.db_url is None and s.device_id == "d9"


def test_example_config_loads(monkeypatch):
    from pathlib import Path
    s = load_settings(Path(__file__).parent.parent / "config" / "example.json")
    assert s.model.backend == "none"


def test_missing_model_file_is_unavailable(tmp_path):
    b, sha = load_model(cfg(path=str(tmp_path / "nope.onnx")))
    assert isinstance(b, UnavailableModel) and sha is None


def test_uniform_multiclass_output_is_uncertain_not_fire():
    # uniform over 4 classes -> P(fire)=0.75 >= 0.70, but the model is uninformative
    a = decide(interpret(softmax_out([0.25, 0.25, 0.25, 0.25]), OutputSpec()), cfg(), STATS, [], "ir")
    assert a.fire_probability == pytest.approx(0.75)
    assert a.status == "uncertain" and a.fire_present is None


def test_presence_only_softmax_model_reports_no_severity():
    spec = OutputSpec(type="softmax_classes", labels=["no_fire", "fire"], no_fire_label="no_fire")
    a = decide(interpret(softmax_out([0.05, 0.95]), spec), cfg(output=spec), STATS, [], "ir")
    assert a.fire_present is True and a.severity_class is None
