"""Single-axis PID with integral clamp and derivative on measurement."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PidGains:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    i_limit: float = 50.0
    out_limit: float = 200.0


class PID:
    def __init__(self, gains: PidGains | None = None):
        self.gains = gains or PidGains()
        self._integral = 0.0
        self._prev_meas: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_meas = None

    def step(self, setpoint: float, measurement: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        err = setpoint - measurement
        self._integral = max(-self.gains.i_limit, min(self.gains.i_limit, self._integral + err * dt))
        if self._prev_meas is None:
            d_meas = 0.0
        else:
            d_meas = (measurement - self._prev_meas) / dt
        self._prev_meas = measurement
        # Derivative on measurement avoids setpoint kicks; sign flips so increasing meas reduces output.
        out = self.gains.kp * err + self.gains.ki * self._integral - self.gains.kd * d_meas
        return max(-self.gains.out_limit, min(self.gains.out_limit, out))
