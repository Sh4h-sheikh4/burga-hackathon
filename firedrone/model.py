"""Model backends + output interpretation.

The pipeline never invents a prediction: if no valid model artifact is configured or it fails to load,
`load_model` returns an UnavailableModel and every frame gets status='model_unavailable', fire_present=NULL.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import ModelConfig, OutputSpec


class Backend(Protocol):
    def run(self, x: np.ndarray) -> list[np.ndarray]: ...


class UnavailableModel:
    def __init__(self, reason: str):
        self.reason = reason

    def run(self, x):  # pragma: no cover - never called by the worker
        raise RuntimeError(self.reason)


class OnnxBackend:
    def __init__(self, path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def run(self, x):
        return [np.asarray(o) for o in self.sess.run(None, {self.input_name: x})]


class TfliteBackend:
    """Recommended on the Pi 3 itself (pip install tflite-runtime)."""
    def __init__(self, path: str):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:  # full TensorFlow fallback
            from tensorflow.lite import Interpreter  # type: ignore
        self.it = Interpreter(model_path=path)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.outs = self.it.get_output_details()

    def run(self, x):
        self.it.set_tensor(self.inp["index"], x.astype(self.inp["dtype"]))
        self.it.invoke()
        return [np.asarray(self.it.get_tensor(o["index"])) for o in self.outs]


class KerasBackend:
    """Intended for the server. Full TensorFlow on a Pi 3 (1 GB RAM) is constrained/not recommended and depends on
    a compatible build for your OS/Python; prefer TFLite or ONNX on the Pi."""
    def __init__(self, path: str):
        import tensorflow as tf  # type: ignore
        self.m = tf.keras.models.load_model(path, compile=False)

    def run(self, x):
        out = self.m.predict(x, verbose=0)
        return [np.asarray(o) for o in (out if isinstance(out, (list, tuple)) else [out])]


class TorchScriptBackend:
    def __init__(self, path: str):
        import torch  # type: ignore
        self.torch = torch
        self.m = torch.jit.load(path, map_location="cpu").eval()

    def run(self, x):
        with self.torch.no_grad():
            out = self.m(self.torch.from_numpy(x))
        out = out if isinstance(out, (list, tuple)) else [out]
        return [o.numpy() for o in out]


BACKENDS = {"onnx": OnnxBackend, "tflite": TfliteBackend, "keras": KerasBackend, "torchscript": TorchScriptBackend}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(cfg: ModelConfig) -> tuple[Backend | UnavailableModel, str | None]:
    if cfg.backend == "none" or not cfg.path:
        return UnavailableModel("no model artifact configured (model.backend/model.path)"), None
    p = Path(cfg.path)
    if not p.exists():
        return UnavailableModel(f"model file not found: {p}"), None
    if cfg.backend not in BACKENDS:
        return UnavailableModel(f"unknown backend {cfg.backend}"), None
    try:
        return BACKENDS[cfg.backend](str(p)), file_sha256(str(p))
    except Exception as e:  # missing runtime, corrupt file, ...
        return UnavailableModel(f"failed to load {cfg.backend} model: {e!r}"), None


def _softmax(v):
    v = np.asarray(v, dtype=np.float64)
    e = np.exp(v - v.max())
    return e / e.sum()


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.asarray(v, dtype=np.float64)))


@dataclass
class RawPrediction:
    fire_probability: float
    severity_probs: dict[str, float] | None      # None when the model has no severity output
    class_probabilities: dict[str, float] = field(default_factory=dict)


def interpret(outputs: list[np.ndarray], spec: OutputSpec) -> RawPrediction:
    """Map raw tensors to P(fire) and P(severity | fire) according to the configured output contract."""
    if spec.type == "binary_sigmoid":
        v = np.ravel(outputs[0])
        if v.size != 1:
            raise ValueError(f"binary_sigmoid expects 1 value, got {v.size}")
        p = float(_sigmoid(v)[0] if spec.apply_sigmoid else v[0])
        return RawPrediction(p, None, {"fire": p})

    if spec.type == "softmax_classes":
        v = np.ravel(outputs[0])
        if v.size != len(spec.labels):
            raise ValueError(f"model emits {v.size} classes but config lists {len(spec.labels)} labels")
        probs = _softmax(v) if spec.apply_softmax else v.astype(np.float64)
        cp = {lab: float(p) for lab, p in zip(spec.labels, probs)}
        p_no = cp[spec.no_fire_label]
        p_fire = float(np.clip(1.0 - p_no, 0.0, 1.0))
        sev = {lab: p for lab, p in cp.items() if lab != spec.no_fire_label}
        tot = sum(sev.values())
        sev = {k: (v / tot if tot > 0 else 0.0) for k, v in sev.items()} if len(sev) > 1 else None
        # (exactly one fire label, e.g. ["no_fire","fire"] = presence-only model -> no severity output)
        return RawPrediction(p_fire, sev, cp)

    if spec.type == "two_head":
        f = np.ravel(outputs[spec.fire_head])
        if f.size == 1:
            p_fire = float(_sigmoid(f)[0] if spec.apply_sigmoid else f[0])
        elif f.size == 2:
            p_fire = float((_softmax(f) if spec.apply_softmax else f)[1])
        else:
            raise ValueError("fire head must have 1 or 2 values")
        s = np.ravel(outputs[spec.severity_head])
        if s.size != len(spec.severity_labels):
            raise ValueError("severity head size != len(severity_labels)")
        sp = _softmax(s) if spec.apply_softmax else s
        sev = {lab: float(p) for lab, p in zip(spec.severity_labels, sp)}
        return RawPrediction(p_fire, sev, {"fire": p_fire, **{f"severity:{k}": v for k, v in sev.items()}})

    raise ValueError(spec.type)
