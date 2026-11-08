# PCB — Raspberry Pi IMU HAT

KiCad 10 project for the hackathon **Pi HAT** with an **LSM6DS3TR-C** IMU on **SPI**.

## Open in KiCad

- `Hackathon_Fall26.kicad_pro` — project  
- `Hackathon_Fall26.kicad_sch` — schematic  
- `Hackathon_Fall26.kicad_pcb` — board  

Renders (top/bottom/copper/schematic) are in [`docs/renders/`](../docs/renders/).

## Notes for judges

- Designed as a Pi HAT; the IMU does **not** back-power the Pi (3V3 only from the Pi).  
- Pipeline software expects the IMU on **SPI** (see `firedrone` IMU config).  
- Power the Pi from a separate **5 V UBEC** off the 4S pack — see [`docs/HARDWARE.md`](../docs/HARDWARE.md).
