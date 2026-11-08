"""JSON config for the attitude stabilize loop (standalone; not part of firedrone)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class StabilizeConfig:
    """Upright attitude hold. Pin map is for off-HAT ESC wiring (HAT has no motor drivers)."""
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
    mix_scale: float = 0.002
    throttle_min: float = 0.0
    throttle_max: float = 0.70
    max_tilt_deg: float = 45.0
    roll_offset_deg: float = 0.0
    pitch_offset_deg: float = 0.0
    imu_interface: str = "spi"           # Hackathon_Fall26 HAT: SPI0/CE0
    imu_spi_bus: int = 0
    imu_spi_device: int = 0
    imu_i2c_bus: int = 1
    imu_address: int = 0x6A


def _from_dict(data: dict[str, Any]) -> StabilizeConfig:
    allowed = {f.name for f in fields(StabilizeConfig)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown stabilize config keys: {sorted(unknown)}")
    return StabilizeConfig(**data)


def load_config(path: str | os.PathLike | None = None) -> StabilizeConfig:
    """Load StabilizeConfig from JSON.

    Accepts either a flat stabilize object, or a wrapper with a top-level \"stabilize\" key
    (and optional top-level imu_* fields used as defaults).
    """
    path = path or os.environ.get("PID_CONTROLLER_CONFIG")
    if not path:
        return StabilizeConfig()
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("config root must be a JSON object")
    if "stabilize" in raw and isinstance(raw["stabilize"], dict):
        body = dict(raw["stabilize"])
        for key in ("imu_interface", "imu_spi_bus", "imu_spi_device", "imu_i2c_bus", "imu_address"):
            if key not in body and key in raw:
                body[key] = raw[key]
        # ignore fire-drone-only keys if someone points at the shared example
        body.pop("enabled", None)
        return validate_stabilize_config(_from_dict(body))
    body = dict(raw)
    body.pop("enabled", None)
    return validate_stabilize_config(_from_dict(body))


def validate_stabilize_config(c: StabilizeConfig) -> StabilizeConfig:
    if c.motor_backend not in ("mock", "pigpio"):
        raise ValueError("motor_backend must be 'mock' or 'pigpio'")
    if len(c.motor_pins_bcm) != 4:
        raise ValueError("motor_pins_bcm must list exactly 4 BCM GPIO pins (FR, RL, FL, RR)")
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
    return c
