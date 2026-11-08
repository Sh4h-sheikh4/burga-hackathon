# Burga Hackathon — Fire-spotting Pi quad

End-to-end prototype for a **5″ X-frame drone** that carries a Raspberry Pi, IMU HAT, and USB camera, uploads frames to **TigerData**, and runs a fire-assessment model.

Built for hackathon judges: open the folders below top-to-bottom.

---

## What’s in the repo

| Folder | What it is |
|---|---|
| [`cad/`](cad/) | SolidWorks **finished drone assembly** (frame + motors + Pi + HAT + battery + camera) |
| [`pcb/`](pcb/) | KiCad **Pi HAT** (LSM6DS3TR-C IMU over SPI) |
| [`pid-controller/`](pid-controller/) | Pi **attitude PID** → X-quad mixer → ESC PWM |
| [`firedrone/`](firedrone/) | Capture / upload / TigerData worker pipeline |
| [`model/`](model/) | CNN trainer + predict CLI for fire presence + severity |
| [`demo_artifacts/`](demo_artifacts/) | Pretrained **smoke-demo** model weights |
| [`demo_images/`](demo_images/) | Tiny synthetic image set for a quick demo |
| [`docs/`](docs/) | Hardware BOM / thrust notes + PCB renders |
| [`config/`](config/) · [`sql/`](sql/) · [`deploy/`](deploy/) | Runtime config, DB schema, systemd units |
| [`tests/`](tests/) · [`tools/`](tools/) | Pytest suite and helper scripts |

---

## Quick story for judges

1. **Airframe** — 5″ X-frame CAD sized for ~800 g AUW ([`cad/`](cad/), [`docs/HARDWARE.md`](docs/HARDWARE.md)).
2. **Sense** — custom HAT reads the IMU over SPI ([`pcb/`](pcb/)).
3. **Fly (software)** — complementary filter + roll/pitch PID on the Pi ([`pid-controller/`](pid-controller/)).
4. **See fire** — USB frames → TigerData → CNN worker → assessments ([`firedrone/`](firedrone/), [`model/`](model/)).

---

## Fastest laptop demo (no Pi required)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements-laptop.txt
# Install a CPU PyTorch build for your platform, then:

python tools/make_smoke_dataset.py
python model/train_predict.py train --data smoke_data --out demo_artifacts --domain rgb --epochs 8 --size 64
```

Full TigerData laptop walkthrough: [`HACKATHON_LAPTOP.md`](HACKATHON_LAPTOP.md).  
Copy `.env.example` → `.env` and set `FIREDRONE_DB_URL` (never commit real secrets).

---

## Hardware snapshot (5″ / 4S)

| Role | Part |
|---|---|
| Motors ×4 | iFlight XING-E Pro 2207 2450KV |
| ESC | SpeedyBee BLS 50A 30×30 |
| Battery | CNHL Black 4S 2200 mAh |
| Compute | Raspberry Pi + Hackathon IMU HAT |
| Pi power | 5 V ≥3 A UBEC from the 4S pack |
| Design AUW | ~800 g → peak TWR ~7:1 |

Details: [`docs/HARDWARE.md`](docs/HARDWARE.md).

---

## Safety

First motor tests **props off or tethered**. The PID loop starts **disarmed**. Never put motor voltage on Pi GPIO.
