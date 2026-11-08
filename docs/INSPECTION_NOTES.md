# PCB & model notes (short)

## IMU HAT (`pcb/`)

- KiCad project `Hackathon_Fall26` — LSM6DS3TR-C on **SPI**.
- HAT draws **3V3 from the Pi only** (does not back-power the Pi).
- Pipeline default: `imu_interface: "spi"` in firedrone config.

## Fire model (`model/train_predict.py`)

- CNN trainer for **fire presence + severity**, exports into the TigerData worker.
- Demo weights: [`demo_artifacts/`](../demo_artifacts/).
- No claim of real-world fire accuracy from synthetic smoke images alone.

## Camera

Described as a USB IR/UVC module; treat brightness stats as relative unless the camera is confirmed radiometric.
