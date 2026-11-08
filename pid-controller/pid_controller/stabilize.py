"""Attitude stabilize loop: IMU -> complementary filter -> roll/pitch PID -> X-quad ESC PWM.

SAFETY (read before enabling motors):
  - Default is disarmed + mock motors. First bench tests must be props-off or the craft tethered.
  - Pass --arm only when ready; Ctrl-C / SIGTERM always disarms and stops pulses.
  - Throttle is clamped to [throttle_min, throttle_max]; tilt above max_tilt_deg forces disarm.

    # From repo root:
    PYTHONPATH=pid-controller python -m pid_controller.stabilize \\
      --config pid-controller/config/example.json --seconds 2

    # On the Pi (props OFF):
    PYTHONPATH=pid-controller python -m pid_controller.stabilize \\
      --config pid-controller/config/example.json --backend pigpio --throttle 0.15 --arm
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass, field

from .attitude import ComplementaryFilter
from .config import StabilizeConfig, load_config
from .imu import ImuSample, open_imu
from .motors import EscPwmConfig, MotorCommand, MotorDriver, command_to_pulses, mix_x_quad, open_motors
from .pid import PID, PidGains

log = logging.getLogger("pid_controller.stabilize")


@dataclass
class StabilizeState:
    armed: bool = False
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_cmd: float = 0.0
    pitch_cmd: float = 0.0
    motors: MotorCommand = field(default_factory=lambda: MotorCommand((0.0, 0.0, 0.0, 0.0)))
    pulses_us: tuple[int, int, int, int] = (1000, 1000, 1000, 1000)
    killed: bool = False
    kill_reason: str = ""


class AttitudeStabilizer:
    """One control tick: estimate attitude, run PIDs, mix, write PWM (or hold idle if disarmed)."""

    def __init__(self, cfg: StabilizeConfig, motors: MotorDriver):
        self.cfg = cfg
        self.motors = motors
        self.filter = ComplementaryFilter(cfg.complementary_alpha)
        self.roll_pid = PID(PidGains(cfg.roll_kp, cfg.roll_ki, cfg.roll_kd, cfg.i_limit, cfg.pid_out_limit))
        self.pitch_pid = PID(PidGains(cfg.pitch_kp, cfg.pitch_ki, cfg.pitch_kd, cfg.i_limit, cfg.pid_out_limit))
        self.state = StabilizeState()
        self._esc = EscPwmConfig(
            pins_bcm=tuple(cfg.motor_pins_bcm),  # type: ignore[arg-type]
            frequency_hz=cfg.pwm_frequency_hz,
            pulse_min_us=cfg.pulse_min_us,
            pulse_max_us=cfg.pulse_max_us,
            pulse_idle_us=cfg.pulse_idle_us,
        )

    def arm(self) -> None:
        if self.state.killed:
            raise RuntimeError(f"cannot arm after kill ({self.state.kill_reason})")
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.filter.reset()
        self.state.armed = True
        log.warning("ARMED — props off / tethered only until proven safe")

    def disarm(self, reason: str = "disarm") -> None:
        self.state.armed = False
        self.motors.disarm()
        idle = (self._esc.pulse_idle_us,) * 4
        self.state.pulses_us = idle
        self.state.motors = MotorCommand((0.0, 0.0, 0.0, 0.0))
        log.info("disarmed (%s)", reason)

    def kill(self, reason: str) -> None:
        self.state.killed = True
        self.state.kill_reason = reason
        self.disarm(reason=f"KILL: {reason}")

    def step(self, sample: ImuSample, dt: float, throttle: float) -> StabilizeState:
        att = self.filter.update(sample, dt)
        roll = att.roll_deg + self.cfg.roll_offset_deg
        pitch = att.pitch_deg + self.cfg.pitch_offset_deg
        self.state.roll_deg, self.state.pitch_deg = roll, pitch

        if abs(roll) > self.cfg.max_tilt_deg or abs(pitch) > self.cfg.max_tilt_deg:
            self.kill(f"tilt limit ({roll:.1f}/{pitch:.1f} deg > {self.cfg.max_tilt_deg})")
            return self.state

        if not self.state.armed:
            self.motors.disarm()
            self.state.pulses_us = (self._esc.pulse_idle_us,) * 4
            return self.state

        thr = max(self.cfg.throttle_min, min(self.cfg.throttle_max, throttle))
        roll_out = self.roll_pid.step(0.0, roll, dt)
        pitch_out = self.pitch_pid.step(0.0, pitch, dt)
        scale = self.cfg.mix_scale
        cmd = mix_x_quad(thr, roll_out * scale, pitch_out * scale, yaw=0.0)
        pulses = command_to_pulses(cmd, self._esc)
        self.motors.set_pulses_us(pulses)
        self.state.roll_cmd, self.state.pitch_cmd = roll_out, pitch_out
        self.state.motors, self.state.pulses_us = cmd, pulses
        return self.state


def _synthetic_level_sample() -> ImuSample:
    return ImuSample(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def run_loop(cfg: StabilizeConfig, seconds: float, throttle: float, arm: bool,
             backend: str, imu=None, rate_hz: float | None = None) -> AttitudeStabilizer:
    rate = rate_hz or cfg.loop_hz
    dt = 1.0 / rate
    motors = open_motors(backend, EscPwmConfig(
        pins_bcm=tuple(cfg.motor_pins_bcm),  # type: ignore[arg-type]
        frequency_hz=cfg.pwm_frequency_hz,
        pulse_min_us=cfg.pulse_min_us,
        pulse_max_us=cfg.pulse_max_us,
        pulse_idle_us=cfg.pulse_idle_us,
    ))
    stab = AttitudeStabilizer(cfg, motors)
    stop = {"flag": False}

    def _stop(signum, frame):
        stop["flag"] = True
        stab.kill(f"signal {signum}")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if imu is None:
        try:
            imu = open_imu(cfg.imu_interface, cfg.imu_i2c_bus, cfg.imu_address,
                           cfg.imu_spi_bus, cfg.imu_spi_device)
        except Exception as e:
            if backend == "mock":
                log.warning("IMU unavailable (%s); using synthetic level samples for dry-run", e)
                imu = None
            else:
                motors.close()
                raise

    try:
        if arm:
            stab.arm()
        else:
            log.info("running disarmed (pass --arm to enable motor pulses above idle)")
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end and not stop["flag"] and not stab.state.killed:
            t0 = time.monotonic()
            sample = imu.read() if imu is not None else _synthetic_level_sample()
            st = stab.step(sample, dt, throttle)
            log.info(
                "roll=%6.2f pitch=%6.2f armed=%s pulses=%s",
                st.roll_deg, st.pitch_deg, st.armed, st.pulses_us,
            )
            sleep = dt - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        stab.disarm("loop exit")
        motors.close()
    return stab


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="JSON stabilize config (see pid-controller/config/example.json)")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--throttle", type=float, default=0.0, help="0..1 base throttle while armed")
    ap.add_argument("--arm", action="store_true", help="enable motor output (DANGEROUS with props)")
    ap.add_argument("--backend", choices=("mock", "pigpio"), default=None,
                    help="default: motor_backend from config (usually mock)")
    ap.add_argument("--rate", type=float, default=None, help="control loop Hz override")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    backend = args.backend or cfg.motor_backend
    if args.arm and backend == "pigpio":
        log.warning("ARM + pigpio: remove props or tether the airframe before continuing")
    run_loop(cfg, args.seconds, args.throttle, args.arm, backend, rate_hz=args.rate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
