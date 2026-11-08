"""Pure decision logic: raw model output + frame quality -> a fire_assessment_v1 record.

Rules (all thresholds configurable in model.thresholds):
  * No model                  -> status 'model_unavailable', fire_present NULL, no severity.
  * P(fire) in the grey zone  -> status 'uncertain', fire_present NULL.
  * Low-quality frame / domain mismatch (if configured) -> downgraded to 'uncertain', fire_present NULL.
  * Severity is reported only when status='ok', fire_present TRUE, the model HAS a severity output, and
    the top severity class probability >= min_severity_confidence. Otherwise severity is NULL.
  * Intensity proxies are always stored separately and never turned into a severity class.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import ModelConfig
from .imaging import FrameStats
from .model import RawPrediction


@dataclass
class Assessment:
    status: str
    fire_present: bool | None = None
    fire_probability: float | None = None
    severity_class: str | None = None
    severity_index: int | None = None
    severity_confidence: float | None = None
    class_probabilities: dict | None = None
    proxy_hot_pixel_fraction: float | None = None
    proxy_max_intensity: float | None = None
    proxy_mean_intensity: float | None = None
    quality_flags: list[str] = field(default_factory=list)
    domain_mismatch: bool = False
    validated: bool = False
    notes: list[str] = field(default_factory=list)


def domain_mismatch(cfg: ModelConfig, frame_domain: str) -> bool:
    return cfg.training_domain != "unknown" and frame_domain != "unknown" and cfg.training_domain != frame_domain


def decide(pred: RawPrediction | None, cfg: ModelConfig, stats: FrameStats | None, flags: list[str],
           frame_domain: str, unavailable_reason: str | None = None) -> Assessment:
    a = Assessment(status="model_unavailable", quality_flags=list(flags))
    if stats is not None:
        a.proxy_hot_pixel_fraction = stats.hot_pixel_fraction
        a.proxy_max_intensity = stats.max_intensity
        a.proxy_mean_intensity = stats.mean_intensity
    a.domain_mismatch = domain_mismatch(cfg, frame_domain)
    if cfg.training_domain == "unknown":
        a.notes.append("model training domain unknown (RGB vs IR not declared)")
    if pred is None:
        a.notes.append(unavailable_reason or "model unavailable")
        return a

    t = cfg.thresholds
    a.fire_probability = pred.fire_probability
    a.class_probabilities = pred.class_probabilities
    a.validated = bool(cfg.validated)
    p = pred.fire_probability
    if p >= t.fire_present_at_or_above:
        a.status, a.fire_present = "ok", True
    elif p <= t.no_fire_at_or_below:
        a.status, a.fire_present = "ok", False
    else:
        a.status, a.fire_present = "uncertain", None
        a.notes.append(f"P(fire)={p:.3f} inside uncertainty band")

    cp = pred.class_probabilities or {}
    if a.status == "ok" and cfg.output.type == "softmax_classes" and len(cp) > 2:
        import math
        v = [max(float(x), 1e-12) for x in cp.values()]
        z = sum(v)
        h = -sum((x / z) * math.log(x / z) for x in v) / math.log(len(v))
        if h > t.max_normalized_entropy:
            # e.g. uniform over 4 classes gives P(fire)=0.75 although the model knows nothing
            a.status, a.fire_present = "uncertain", None
            a.notes.append(f"class distribution nearly uniform (normalized entropy {h:.2f}); output uninformative")
    if a.status == "ok" and flags and cfg.treat_low_quality_as_uncertain:
        a.status, a.fire_present = "uncertain", None
        a.notes.append("low-quality frame: " + ",".join(flags))
    if a.status == "ok" and a.domain_mismatch and cfg.treat_domain_mismatch_as_uncertain:
        a.status, a.fire_present = "uncertain", None
        a.notes.append(f"domain mismatch: model trained on {cfg.training_domain}, frame is {frame_domain}")

    if a.status == "ok" and a.fire_present:
        if pred.severity_probs is None:
            a.notes.append("model has no severity output; severity withheld")
        else:
            labels = cfg.output.severity_labels if cfg.output.type == "two_head" else \
                [l for l in cfg.output.labels if l != cfg.output.no_fire_label]
            best = max(pred.severity_probs, key=pred.severity_probs.get)
            conf = pred.severity_probs[best]
            a.severity_confidence = float(conf)
            if conf >= t.min_severity_confidence:
                a.severity_class = best
                a.severity_index = labels.index(best)
            else:
                a.notes.append(f"severity confidence {conf:.3f} < {t.min_severity_confidence}; withheld")
    if not cfg.validated:
        a.notes.append("model NOT validated on labeled held-out data")
    return a
