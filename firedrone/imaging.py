"""Image decoding, preprocessing, quality checks and intensity proxies (numpy + Pillow only)."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import InputSpec, Thresholds

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def decode(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    return img


def encode_jpeg(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB" if img.mode not in ("L", "RGB") else img.mode).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def preprocess(img: Image.Image, spec: InputSpec) -> np.ndarray:
    """Return a batch of 1 shaped for the model. Grayscale IR is replicated to 3 channels if needed."""
    img = img.convert("L" if spec.channels == 1 else "RGB").resize((spec.width, spec.height), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        x = x[..., None]
    if spec.scale == "0_1":
        x = x / 255.0
    elif spec.scale == "minus1_1":
        x = x / 127.5 - 1.0
    elif spec.scale == "imagenet":
        if x.shape[-1] != 3:
            raise ValueError("imagenet normalisation needs 3 channels")
        x = (x / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    if spec.layout == "NCHW":
        x = np.transpose(x, (2, 0, 1))
    return x[None].astype(spec.dtype)


@dataclass
class FrameStats:
    mean_intensity: float
    max_intensity: float
    saturated_fraction: float
    hot_pixel_fraction: float   # fraction of very bright pixels; NOT temperature unless camera is calibrated
    sharpness: float


def frame_stats(img: Image.Image, t: Thresholds) -> FrameStats:
    g = np.asarray(img.convert("L"), dtype=np.float32)
    # Laplacian variance as a cheap blur metric.
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return FrameStats(
        mean_intensity=float(g.mean()),
        max_intensity=float(g.max()),
        saturated_fraction=float((g >= 254).mean()),
        hot_pixel_fraction=float((g >= t.proxy_hot_level).mean()),
        sharpness=float(lap.var()) if lap.size else 0.0,
    )


def quality_flags(stats: FrameStats, t: Thresholds, gyro_dps: float | None = None) -> list[str]:
    flags = []
    if stats.mean_intensity < t.min_mean_intensity:
        flags.append("too_dark")
    if stats.saturated_fraction > t.max_saturated_fraction:
        flags.append("saturated")
    if stats.sharpness < t.min_sharpness:
        flags.append("blurry_or_flat")
    if gyro_dps is not None and gyro_dps > t.max_gyro_dps:
        flags.append("high_rotation_rate")
    return flags
