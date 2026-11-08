"""Configuration loading. Everything is driven by a JSON file plus environment variables.

Secrets (the database URL) come ONLY from the environment (FIREDRONE_DB_URL), never from the JSON file.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class InputSpec:
    width: int = 224
    height: int = 224
    channels: int = 3            # 1 = grayscale, 3 = RGB (grayscale IR is replicated to 3 channels)
    layout: str = "NHWC"         # "NHWC" (Keras/TFLite) or "NCHW" (PyTorch/ONNX-from-torch)
    scale: str = "0_1"           # "0_1", "minus1_1", "imagenet", "raw_0_255"
    dtype: str = "float32"


@dataclass
class OutputSpec:
    # "binary_sigmoid": one value = P(fire). Model provides NO severity.
    # "softmax_classes": one vector over `labels`; `no_fire_label` marks the negative class.
    # "two_head": outputs[fire_head] = P(fire) (1 value or 2-way softmax), outputs[severity_head] = softmax over severity_labels.
    type: str = "softmax_classes"
    labels: list[str] = field(default_factory=lambda: ["no_fire", "low", "moderate", "high"])
    no_fire_label: str = "no_fire"
    severity_labels: list[str] = field(default_factory=lambda: ["low", "moderate", "high"])
    fire_head: int = 0
    severity_head: int = 1
    apply_softmax: bool = False  # set True if the model outputs logits
    apply_sigmoid: bool = False


@dataclass
class Thresholds:
    fire_present_at_or_above: float = 0.70   # P(fire) >= this -> fire_present TRUE
    no_fire_at_or_below: float = 0.30        # P(fire) <= this -> fire_present FALSE; in between -> uncertain/NULL
    min_severity_confidence: float = 0.50    # below this the severity class is withheld (NULL)
    max_normalized_entropy: float = 0.90     # class distribution flatter than this = uninformative -> uncertain
    min_mean_intensity: float = 5.0          # frame almost black -> quality flag
    max_saturated_fraction: float = 0.60     # most pixels clipped -> quality flag
    min_sharpness: float = 2.0               # very blurry (motion) -> quality flag
    max_gyro_dps: float = 120.0              # IMU says the drone was rotating fast -> motion-blur flag
    proxy_hot_level: int = 240               # 8-bit level counted as "very bright" (not a temperature) for the proxy


@dataclass
class ModelConfig:
    name: str = "unconfigured"
    version: str = "0"
    backend: str = "none"        # "onnx", "tflite", "keras", "torchscript", "none"
    path: str | None = None
    training_domain: str = "unknown"  # "rgb", "ir", "unknown" - what the training images were. NOTE: "ir" does not
                                      # distinguish near-IR from thermal; validate on frames from the same camera type.
    validated: bool = False           # set TRUE only after tools/evaluate.py on a labeled held-out set
    treat_domain_mismatch_as_uncertain: bool = True
    treat_low_quality_as_uncertain: bool = True
    input: InputSpec = field(default_factory=InputSpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    thresholds: Thresholds = field(default_factory=Thresholds)


@dataclass
class StabilizeConfig:
    """Upright attitude hold (see firedrone.stabilize). Pin map is for off-HAT ESC wiring."""
    enabled: bool = False
    motor_backend: str = "mock"          # "mock" (safe default) or "pigpio"
    motor_pins_bcm: list[int] = field(default_factory=lambda: [12, 13, 18, 19])  # FR, RL, FL, RR
    pwm_frequency_hz: int = 50
    pulse_min_us: int = 1000
    pulse_max_us: int = 2000
    pulse_idle_us: int = 1000
    loop_hz: float = 50.0
    complementary_alpha: float = 0.98
    roll_kp: float = 1.5
    roll_ki: float = 0.0
    roll_kd: float = 0.05
    pitch_kp: float = 1.5
    pitch_ki: float = 0.0
    pitch_kd: float = 0.05
    i_limit: float = 30.0
    pid_out_limit: float = 200.0
    mix_scale: float = 0.002             # maps PID output -> mixer units
    throttle_min: float = 0.0
    throttle_max: float = 0.70           # hard ceiling; raise only after tethered tests
    max_tilt_deg: float = 45.0           # auto-kill if exceeded
    roll_offset_deg: float = 0.0
    pitch_offset_deg: float = 0.0
    # IMU path for the stabilize process (mirrors Settings IMU defaults / SPI HAT wiring)
    imu_interface: str = "spi"
    imu_spi_bus: int = 0
    imu_spi_device: int = 0
    imu_i2c_bus: int = 1
    imu_address: int = 0x6A


@dataclass
class Settings:
    db_url: str | None
    device_id: str = "drone-01"
    mission_id: str | None = None
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    capture_interval_s: float = 1.0
    jpeg_quality: int = 85
    sensor_domain: str = "ir"            # user said "USB IR camera"; exact type (near-IR vs thermal) unknown
    spool_dir: str = "./spool"
    imu_enabled: bool = True
    imu_interface: str = "spi"         # Hackathon_Fall26 HAT wires the IMU to SPI0/CE0 (see docs/INSPECTION_NOTES.md)
    imu_spi_bus: int = 0
    imu_spi_device: int = 0            # CE0 = GPIO8 = header pin 24
    imu_i2c_bus: int = 1
    imu_address: int = 0x6A           # 0x6A if SDO/SA0 tied to GND, 0x6B if tied to VDDIO
    worker_batch_size: int = 8
    worker_poll_s: float = 1.0
    worker_max_attempts: int = 3
    worker_stale_claim_s: int = 300
    model: ModelConfig = field(default_factory=ModelConfig)
    stabilize: StabilizeConfig = field(default_factory=StabilizeConfig)


def _merge(dc_cls, data: dict[str, Any]):
    kwargs = {}
    for k, v in (data or {}).items():
        f = dc_cls.__dataclass_fields__.get(k)
        if f is None:
            raise ValueError(f"Unknown config key '{k}' for {dc_cls.__name__}")
        sub = {
            "input": InputSpec,
            "output": OutputSpec,
            "thresholds": Thresholds,
            "model": ModelConfig,
            "stabilize": StabilizeConfig,
        }.get(k)
        kwargs[k] = _merge(sub, v) if sub and isinstance(v, dict) else v
    return dc_cls(**kwargs)


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    path = path or os.environ.get("FIREDRONE_CONFIG")
    data: dict[str, Any] = {}
    if path:
        data = json.loads(Path(path).read_text())
    data.pop("db_url", None)  # never read secrets from the file
    s = _merge(Settings, {"db_url": None, **data})
    s.db_url = os.environ.get("FIREDRONE_DB_URL")
    if os.environ.get("FIREDRONE_DEVICE_ID"):
        s.device_id = os.environ["FIREDRONE_DEVICE_ID"]
    if os.environ.get("FIREDRONE_MISSION_ID"):
        s.mission_id = os.environ["FIREDRONE_MISSION_ID"]
    if path and s.model.path and not Path(s.model.path).is_absolute():
        s.model.path = str((Path(path).resolve().parent / s.model.path).resolve())  # relative to the config file
    # Stabilize reuses top-level IMU wiring unless the stabilize block overrides it.
    if "stabilize" not in data or "imu_interface" not in (data.get("stabilize") or {}):
        s.stabilize.imu_interface = s.imu_interface
        s.stabilize.imu_spi_bus = s.imu_spi_bus
        s.stabilize.imu_spi_device = s.imu_spi_device
        s.stabilize.imu_i2c_bus = s.imu_i2c_bus
        s.stabilize.imu_address = s.imu_address
    validate_model_config(s.model)
    validate_stabilize_config(s.stabilize)
    return s


def validate_stabilize_config(c: StabilizeConfig) -> None:
    if c.motor_backend not in ("mock", "pigpio"):
        raise ValueError("stabilize.motor_backend must be 'mock' or 'pigpio'")
    if len(c.motor_pins_bcm) != 4:
        raise ValueError("stabilize.motor_pins_bcm must list exactly 4 BCM GPIO pins (FR, RL, FL, RR)")
    if not (0.0 <= c.throttle_min <= c.throttle_max <= 1.0):
        raise ValueError("need 0 <= throttle_min <= throttle_max <= 1")
    if not (0.0 <= c.complementary_alpha <= 1.0):
        raise ValueError("complementary_alpha must be in [0, 1]")
    if c.pulse_min_us >= c.pulse_max_us:
        raise ValueError("pulse_min_us must be < pulse_max_us")
    if c.loop_hz <= 0:
        raise ValueError("loop_hz must be > 0")
    if c.max_tilt_deg <= 0:
        raise ValueError("max_tilt_deg must be > 0")


def validate_model_config(m: ModelConfig) -> None:
    t = m.thresholds
    if not (0.0 <= t.no_fire_at_or_below <= t.fire_present_at_or_above <= 1.0):
        raise ValueError("Need 0 <= no_fire_at_or_below <= fire_present_at_or_above <= 1")
    o = m.output
    if o.type not in ("binary_sigmoid", "softmax_classes", "two_head"):
        raise ValueError(f"Unsupported output.type {o.type}")
    if o.type == "softmax_classes" and o.no_fire_label not in o.labels:
        raise ValueError("softmax_classes needs no_fire_label inside labels")
    if m.input.layout not in ("NHWC", "NCHW"):
        raise ValueError("input.layout must be NHWC or NCHW")
    if m.input.scale not in ("0_1", "minus1_1", "imagenet", "raw_0_255"):
        raise ValueError("bad input.scale")


def config_sha256(m: ModelConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(m), sort_keys=True).encode()).hexdigest()
