"""Quad ESC PWM outputs.

Uses pigpio hardware-timed pulses when available (preferred on a Pi). Falls back to a
recording MockMotors for laptop/unit tests. Pulse widths are microseconds (typical
ESC range 1000–2000 µs at ~50 Hz).

Default BCM pin map (override in config — HAT has no motor drivers; wire ESCs yourself):
  M0 front-right  GPIO 12 (header pin 32)
  M1 rear-left    GPIO 13 (header pin 33)
  M2 front-left   GPIO 18 (header pin 12)
  M3 rear-right   GPIO 19 (header pin 35)

Avoid SPI0 pins used by the Hackathon_Fall26 IMU (GPIO 8/9/10/11) and ID EEPROM I2C
(GPIO 0/1 on the ID bus; GPIO 2/3 if you also use that bus).
"""
from __future__ import annotations

from dataclasses import dataclass, field


# X-quad indices used everywhere: FR, RL, FL, RR
MOTOR_LABELS = ("front_right", "rear_left", "front_left", "rear_right")
DEFAULT_MOTOR_PINS_BCM = (12, 13, 18, 19)


@dataclass
class EscPwmConfig:
    pins_bcm: tuple[int, int, int, int] = DEFAULT_MOTOR_PINS_BCM
    frequency_hz: int = 50
    pulse_min_us: int = 1000   # ESC idle / disarmed
    pulse_max_us: int = 2000
    pulse_idle_us: int = 1000


@dataclass
class MotorCommand:
    """Normalized thrust mix before ESC mapping: 0.0 = idle, 1.0 = full."""
    values: tuple[float, float, float, float]  # FR, RL, FL, RR


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def mix_x_quad(throttle: float, roll: float, pitch: float, yaw: float = 0.0) -> MotorCommand:
    """Classic X mixer. Positive roll = more left thrust; positive pitch = nose-up (more rear);
    positive yaw = more CCW motors (FL+RR). Inputs are dimensionless mix units, not degrees."""
    fr = throttle - roll - pitch - yaw
    rl = throttle + roll + pitch - yaw
    fl = throttle + roll - pitch + yaw
    rr = throttle - roll + pitch + yaw
    return MotorCommand((fr, rl, fl, rr))


def command_to_pulses(cmd: MotorCommand, cfg: EscPwmConfig) -> tuple[int, int, int, int]:
    span = cfg.pulse_max_us - cfg.pulse_min_us
    out = []
    for v in cmd.values:
        us = cfg.pulse_min_us + int(round(clamp01(v) * span))
        out.append(max(cfg.pulse_min_us, min(cfg.pulse_max_us, us)))
    return tuple(out)  # type: ignore[return-value]


class MotorDriver:
    """Write four ESC pulse widths (µs). Disarmed drivers must force idle/min."""

    def set_pulses_us(self, pulses: tuple[int, int, int, int]) -> None:
        raise NotImplementedError

    def disarm(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


@dataclass
class MockMotors(MotorDriver):
    """No GPIO — records the last pulse command for tests / dry-run."""
    cfg: EscPwmConfig = field(default_factory=EscPwmConfig)
    last_pulses_us: tuple[int, int, int, int] | None = None
    history: list[tuple[int, int, int, int]] = field(default_factory=list)

    def set_pulses_us(self, pulses: tuple[int, int, int, int]) -> None:
        self.last_pulses_us = pulses
        self.history.append(pulses)

    def disarm(self) -> None:
        idle = (self.cfg.pulse_idle_us,) * 4
        self.set_pulses_us(idle)


class PigpioMotors(MotorDriver):
    """ESC PWM via pigpio (sudo apt install pigpio python3-pigpio; sudo systemctl enable --now pigpiod)."""

    def __init__(self, cfg: EscPwmConfig, pi=None):
        self.cfg = cfg
        if pi is None:
            import pigpio  # type: ignore
            pi = pigpio.pi()
            if not pi.connected:
                raise RuntimeError("pigpio daemon not running (start with: sudo systemctl start pigpiod)")
        self.pi = pi
        self._own_pi = pi is not None
        for pin in cfg.pins_bcm:
            self.pi.set_mode(pin, 1)  # OUTPUT
            self.pi.set_PWM_frequency(pin, cfg.frequency_hz)
            # pigpio set_servo_pulsewidth is the standard ESC path (50 Hz hardware-timed).
            self.pi.set_servo_pulsewidth(pin, cfg.pulse_idle_us)

    def set_pulses_us(self, pulses: tuple[int, int, int, int]) -> None:
        for pin, us in zip(self.cfg.pins_bcm, pulses):
            us = max(self.cfg.pulse_min_us, min(self.cfg.pulse_max_us, int(us)))
            self.pi.set_servo_pulsewidth(pin, us)

    def disarm(self) -> None:
        for pin in self.cfg.pins_bcm:
            self.pi.set_servo_pulsewidth(pin, self.cfg.pulse_idle_us)

    def close(self) -> None:
        try:
            self.disarm()
            for pin in self.cfg.pins_bcm:
                self.pi.set_servo_pulsewidth(pin, 0)  # stop pulses
        finally:
            if hasattr(self.pi, "stop"):
                self.pi.stop()


def open_motors(backend: str, cfg: EscPwmConfig) -> MotorDriver:
    if backend == "mock":
        return MockMotors(cfg)
    if backend == "pigpio":
        return PigpioMotors(cfg)
    raise ValueError(f"motor_backend must be 'mock' or 'pigpio', got {backend!r}")
