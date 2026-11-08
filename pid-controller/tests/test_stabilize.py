"""Unit tests for attitude estimate, PID, mixer, and arming/safety (no GPIO)."""
from pathlib import Path

import pytest

from pid_controller.attitude import ComplementaryFilter, accel_roll_pitch_deg
from pid_controller.config import StabilizeConfig, load_config, validate_stabilize_config
from pid_controller.imu import ImuSample
from pid_controller.motors import MockMotors, EscPwmConfig, command_to_pulses, mix_x_quad, open_motors
from pid_controller.pid import PID, PidGains
from pid_controller.stabilize import AttitudeStabilizer

ROOT = Path(__file__).resolve().parents[1]


def test_accel_level_is_zero_tilt():
    a = accel_roll_pitch_deg(ImuSample(0, 0, 0, 0, 0, 1))
    assert abs(a.roll_deg) < 1e-6 and abs(a.pitch_deg) < 1e-6


def test_accel_roll_right_wing_down():
    a = accel_roll_pitch_deg(ImuSample(0, 0, 0, 0, 1, 1))
    assert a.roll_deg == pytest.approx(45.0, abs=0.1)


def test_complementary_seeds_from_accel_then_tracks_gyro():
    f = ComplementaryFilter(alpha=1.0)
    f.update(ImuSample(0, 0, 0, 0, 0, 1), 0.01)
    assert f.roll_deg == pytest.approx(0.0)
    for _ in range(100):
        f.update(ImuSample(10.0, 0, 0, 0, 0, 1), 0.01)
    assert f.roll_deg == pytest.approx(10.0, abs=0.05)


def test_pid_proportional_opposes_error():
    pid = PID(PidGains(kp=2.0, ki=0.0, kd=0.0, out_limit=100))
    assert pid.step(0.0, 5.0, 0.02) == pytest.approx(-10.0)


def test_pid_integral_clamped():
    pid = PID(PidGains(kp=0.0, ki=10.0, kd=0.0, i_limit=1.0, out_limit=100))
    for _ in range(50):
        out = pid.step(0.0, -1.0, 0.1)
    assert out == pytest.approx(10.0)


def test_x_mixer_roll_right_boosts_left():
    cmd = mix_x_quad(0.5, roll=0.1, pitch=0.0, yaw=0.0)
    fr, rl, fl, rr = cmd.values
    assert fl > fr and rl > rr


def test_x_mixer_pitch_nose_up_boosts_rear():
    cmd = mix_x_quad(0.5, roll=0.0, pitch=0.1, yaw=0.0)
    fr, rl, fl, rr = cmd.values
    assert rl > fl and rr > fr


def test_pulses_clamp_to_esc_range():
    cfg = EscPwmConfig(pulse_min_us=1000, pulse_max_us=2000)
    pulses = command_to_pulses(mix_x_quad(2.0, 0, 0), cfg)
    assert pulses == (2000, 2000, 2000, 2000)
    pulses = command_to_pulses(mix_x_quad(-1.0, 0, 0), cfg)
    assert pulses == (1000, 1000, 1000, 1000)


def test_open_motors_rejects_bad_backend():
    with pytest.raises(ValueError):
        open_motors("rpi_gpio", EscPwmConfig())


def test_disarmed_holds_idle_even_with_tilt():
    motors = MockMotors()
    stab = AttitudeStabilizer(StabilizeConfig(), motors)
    sample = ImuSample(0, 0, 0, 0.2, 0, 1)
    st = stab.step(sample, 0.02, throttle=0.5)
    assert st.armed is False
    assert st.pulses_us == (1000, 1000, 1000, 1000)
    assert motors.last_pulses_us == (1000, 1000, 1000, 1000)


def test_armed_level_uses_throttle():
    motors = MockMotors()
    cfg = StabilizeConfig(roll_kp=0, pitch_kp=0, roll_kd=0, pitch_kd=0, mix_scale=0.002)
    stab = AttitudeStabilizer(cfg, motors)
    stab.arm()
    st = stab.step(ImuSample(0, 0, 0, 0, 0, 1), 0.02, throttle=0.2)
    assert st.armed and all(1100 <= p <= 1300 for p in st.pulses_us)


def test_tilt_limit_kills():
    motors = MockMotors()
    stab = AttitudeStabilizer(StabilizeConfig(max_tilt_deg=30.0), motors)
    stab.arm()
    st = stab.step(ImuSample(0, 0, 0, 0, 1, 1), 0.02, throttle=0.2)
    assert st.killed and not st.armed
    assert motors.last_pulses_us == (1000, 1000, 1000, 1000)


def test_cannot_rearm_after_kill():
    motors = MockMotors()
    stab = AttitudeStabilizer(StabilizeConfig(max_tilt_deg=10.0), motors)
    stab.arm()
    stab.step(ImuSample(0, 0, 0, 0, 1, 1), 0.02, 0.1)
    with pytest.raises(RuntimeError):
        stab.arm()


def test_config_loads_from_example_json():
    cfg = load_config(ROOT / "config" / "example.json")
    assert cfg.motor_backend == "mock"
    assert cfg.motor_pins_bcm == [12, 13, 18, 19]
    assert cfg.imu_interface == "spi"


def test_stabilize_config_validation():
    with pytest.raises(ValueError):
        validate_stabilize_config(StabilizeConfig(motor_pins_bcm=[1, 2, 3]))
    with pytest.raises(ValueError):
        validate_stabilize_config(StabilizeConfig(throttle_min=0.8, throttle_max=0.2))
