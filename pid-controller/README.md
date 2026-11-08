# PID controller — upright attitude hold

IMU → complementary filter → roll/pitch PID → **X-quad mixer** → ESC PWM.

Package: `pid_controller/`. First hardware tests: **props off or tethered**.

## Hardware (summary)

Full BOM and thrust tables: [`docs/HARDWARE.md`](../docs/HARDWARE.md).

| Role | Part |
|---|---|
| Motors ×4 | iFlight XING-E Pro 2207 2450KV |
| Props | 5″ 5045 / 51466 class (2 CW + 2 CCW) |
| ESC | SpeedyBee BLS 50A 30×30 |
| Battery | CNHL Black 4S 2200 mAh |
| Pi power | 5 V ≥3 A UBEC |

Design AUW ~800 g → peak TWR ~7.3:1.

Default ESC signal pins (BCM): **12, 13, 18, 19** (FR, RL, FL, RR) — override in config.

## Layout

```
pid-controller/
  README.md
  config/example.json
  pid_controller/          # attitude, pid, motors, imu, stabilize
  tests/test_stabilize.py
```

## Run

From the **repo root**:

```bash
# Unit tests (no GPIO / no Pi)
PYTHONPATH=pid-controller python -m pytest pid-controller/tests -q

# Dry-run (mock motors)
PYTHONPATH=pid-controller python -m pid_controller.stabilize \
  --config pid-controller/config/example.json --seconds 2

# On the Pi (props OFF / tethered):
# sudo apt install pigpio python3-pigpio && sudo systemctl enable --now pigpiod
# PYTHONPATH=pid-controller python -m pid_controller.stabilize \
#   --config pid-controller/config/example.json --backend pigpio --throttle 0.15 --arm
```

Safety: default backend is `mock`; loop starts **disarmed**. Ctrl-C disarms. Excess tilt kills output.
