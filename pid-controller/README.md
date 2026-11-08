# PID controller (upright attitude hold)

Self-contained flight-stabilize stack for the burga-hackathon Raspberry Pi quad:

**IMU (gyro + accel) → complementary filter → roll/pitch PID → X-quad mixer → ESC PWM.**

Package path: `pid-controller/pid_controller/`. First hardware tests: **props off or tethered**.

---

## Recommended motors and battery

Source of truth: project doc `docs/motors-and-battery.md` (Project store). Summary of the selected 5″ / 4S powertrain:

| Role | Part |
|---|---|
| Motors (×4) | **iFlight XING-E Pro 2207 2450KV** (~33.8 g each; ~1465 g thrust on 5045 @ 4S) |
| Props | **HQProp 5×4.5×3** or **Gemfan Hurricane 51466** / **5045** biprop (2 CW + 2 CCW) |
| ESC | **SpeedyBee BLS 50A 30×30 4-in-1** (3–6S, ~13.8 g) |
| Battery | **CNHL Black 4S 2200 mAh** (~40C, XT60; stock 220404BK) |
| Pi power | **Hobbywing 5V/6V 5A UBEC** or **Matek UBEC Duo** set to **5 V** (≥3 A for Pi 4) |

**AUW / thrust (assumed ~800 g all-up with Pi 4 + HAT + camera + frame):**

- Peak thrust ≈ **5860 g** (4 × 1465 g) → peak TWR ≈ **7.3 : 1**
- Hover ≈ **200 g/motor** (~35–45% throttle typical)
- Hover endurance on 2200 mAh ≈ **4–5 min** (order-of-magnitude)

**Wiring sketch:** 4S LiPo → ESC → motors; same pack → UBEC 5 V → Pi USB-C. HAT draws 3V3 from the Pi only (does not back-power it). Never put motor voltage on GPIO. Default ESC signal pins in software: BCM **12, 13, 18, 19** (FR, RL, FL, RR) — override in config.

Acceptable motor alternate: EMAX ECO II 2207 ~2400KV on 4S (do not mix a 6S motor/pack choice without re-sizing ESC/props).

---

## Layout

```
pid-controller/
  README.md                 # this file
  config/example.json       # PWM pins, PID gains, IMU interface
  pid_controller/           # Python package
    attitude.py             # complementary filter
    pid.py                  # PID
    motors.py               # X-mixer + mock/pigpio ESC PWM
    imu.py                  # LSM6DS3TR-C (SPI default, same HAT wiring as firedrone)
    stabilize.py            # arm/disarm/kill control loop CLI
    config.py
  tests/test_stabilize.py
```

---

## How to run

From the **repo root** (`burga-hackathon/`):

```bash
# Unit tests (no GPIO / no Pi required)
PYTHONPATH=pid-controller python3 -m pytest pid-controller/tests -q

# Dry-run (mock motors, synthetic IMU if no hardware)
PYTHONPATH=pid-controller python3 -m pid_controller.stabilize \
  --config pid-controller/config/example.json --seconds 2

# On the Pi after: sudo apt install pigpio python3-pigpio && sudo systemctl enable --now pigpiod
# PROPS OFF or tethered:
# PYTHONPATH=pid-controller python3 -m pid_controller.stabilize \
#   --config pid-controller/config/example.json --backend pigpio --throttle 0.15 --arm
```

Safety: default backend is `mock` and the loop starts **disarmed**. `--arm` enables pulses. Ctrl-C / SIGTERM disarms. Tilt beyond `max_tilt_deg` kills output. Tune gains in `config/example.json`.
