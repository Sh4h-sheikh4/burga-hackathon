# CAD — finished 5″ X-frame drone

Open **`FinishedDrone_XFrame.SLDASM`** in SolidWorks (2026 used during design).

All parts live in this folder so references resolve when the folder is kept together.

## Files

| File | Description |
|---|---|
| `FinishedDrone_XFrame.SLDASM` | Full assembly |
| `XFrame_5inch_PiQuad.SLDPRT` | 5″ X-frame (250 mm wheelbase) |
| `Motor_XING-E_Pro_2207.SLDPRT` | Simplified motor envelope (×4 in assembly) |
| `RaspberryPi3.SLDPRT` | Pi 3 board envelope |
| `PiHAT_IMU.SLDPRT` | IMU HAT envelope |
| `Battery_CNHL_4S_2200.SLDPRT` | 4S 2200 mAh pack (underside) |
| `USB_Camera.SLDPRT` | Upright USB camera (forward-facing) |

## Layout

- **Top:** Raspberry Pi + IMU HAT  
- **Arms:** four XING-E Pro 2207 motors on 16×16 mm pads  
- **Bottom:** CNHL 4S 2200 battery + upright camera looking forward  

Mounting patterns match the BOM in [`docs/HARDWARE.md`](../docs/HARDWARE.md):

- Motors: 16×16 mm M3  
- ESC stack (on frame): 30.5×30.5 mm  
- Pi: 58×49 mm holes  

Frame mass ≈ **85 g** in PLA — under the ~220 g frame budget used for AUW sizing.
