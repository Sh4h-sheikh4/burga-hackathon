"""Roll/pitch estimate from LSM6DS3TR-C gyro + accelerometer (complementary filter).

Gyro axes are in deg/s; accel in g. Convention (aircraft, IMU flat on the HAT):
  roll  = rotation about +X (nose), positive right-wing-down
  pitch = rotation about +Y (right wing), positive nose-up

Axis mapping (gyro_x/y/z, accel_x/y/z from firedrone.imu) assumes the chip X/Y/Z
align with the airframe; remount offsets belong in config (see pid_controller.config.StabilizeConfig).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .imu import ImuSample


@dataclass
class Attitude:
    roll_deg: float
    pitch_deg: float


def accel_roll_pitch_deg(sample: ImuSample) -> Attitude:
    """Tilt from gravity; undefined under strong linear accel / free-fall."""
    ax, ay, az = sample.accel_x, sample.accel_y, sample.accel_z
    roll = math.degrees(math.atan2(ay, az if az != 0 else 1e-9))
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
    return Attitude(roll, pitch)


class ComplementaryFilter:
    """alpha near 1 trusts gyro more; typical 0.95–0.98 at ~50–100 Hz."""

    def __init__(self, alpha: float = 0.98, roll0: float = 0.0, pitch0: float = 0.0):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.roll_deg = roll0
        self.pitch_deg = pitch0
        self._seeded = False

    def reset(self, roll0: float = 0.0, pitch0: float = 0.0) -> None:
        self.roll_deg, self.pitch_deg, self._seeded = roll0, pitch0, False

    def update(self, sample: ImuSample, dt: float) -> Attitude:
        acc = accel_roll_pitch_deg(sample)
        if not self._seeded:
            self.roll_deg, self.pitch_deg = acc.roll_deg, acc.pitch_deg
            self._seeded = True
            return Attitude(self.roll_deg, self.pitch_deg)
        # Integrate body rates (deg/s). Small-angle approx is fine for upright stabilize.
        gyro_roll = self.roll_deg + sample.gyro_x * dt
        gyro_pitch = self.pitch_deg + sample.gyro_y * dt
        a = self.alpha
        self.roll_deg = a * gyro_roll + (1.0 - a) * acc.roll_deg
        self.pitch_deg = a * gyro_pitch + (1.0 - a) * acc.pitch_deg
        return Attitude(self.roll_deg, self.pitch_deg)
