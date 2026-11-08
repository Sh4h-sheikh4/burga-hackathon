# Hardware — motors, battery, and AUW

Propulsion sizing for the Pi quad: custom CAD frame ([`cad/`](../cad/)) + Raspberry Pi + IMU HAT ([`pcb/`](../pcb/)) + USB camera + LiPo.

Flight-control software lives in [`pid-controller/`](../pid-controller/).

---

## Design AUW (~800 g)

| Item | Mass (g) | Notes |
|---|---:|---|
| CAD frame + arms + hardware (5″) | **~85–220** | Printed frame measured ~85 g PLA; keep ≤220 g budget |
| Raspberry Pi (3 or 4) | 45–46 | Board only |
| IMU HAT | 20 | Custom HAT or Sense HAT |
| USB camera + mount | 50 | Compact UVC; weigh the real module |
| 4× iFlight XING-E Pro 2207 | 136 | 33.8 g each with wires |
| 4× 5″ props | 20 | ~5 g each |
| SpeedyBee BLS 50A 4-in-1 | 14 | 30.5×30.5 mm |
| UBEC + wiring / XT60 / straps | 65 | 5 V ≥3 A for the Pi |
| **Dry subtotal** | **~435–570** | |
| CNHL Black 4S 2200 mAh | **230** | ~211–256 g vendor range |
| **Design AUW** | **~800 g** | |

If AUW grows to ~1000 g, peak TWR is still ~5.9:1. Revisit motors only above ~1.4 kg.

---

## Recommended powertrain (5″ / 4S)

| Role | Part |
|---|---|
| Motors ×4 | **iFlight XING-E Pro 2207 2450KV** — 16×16 mm M3; ~1465 g thrust on 5045 @ 4S |
| Props | HQProp 5×4.5×3 or Gemfan Hurricane 51466 / 5045 (2 CW + 2 CCW) |
| ESC | **SpeedyBee BLS 50A 30×30** 4-in-1 (3–6S) |
| Battery | **CNHL Black 4S 2200 mAh** XT60 (~40C; stock 220404BK) |
| Pi power | Hobbywing 5V/6V 5A UBEC or Matek UBEC Duo @ **5 V** |

### Thrust (iFlight bench, 16 V / 5045)

| Throttle | Current | Thrust / motor |
|---:|---:|---:|
| 50% | 6.8 A | 476 g |
| 70% | 14.5 A | 792 g |
| 100% | 32 A | **1465 g** |

At ~800 g AUW: peak thrust ≈ **5860 g** → TWR ≈ **7.3 : 1**. Hover ≈ 200 g/motor (~35–45% throttle). Endurance on 2200 mAh ≈ **4–5 min** hover (order-of-magnitude).

---

## Wiring

```
[4S LiPo XT60]
    ├─► SpeedyBee 4-in-1 ESC  ──► 4× XING-E Pro 2207
    │         ▲ DShot/PWM from Pi GPIO (pid-controller)
    └─► UBEC (5 V, ≥3–5 A) ──► Pi USB-C
              Pi 3V3 ──► IMU HAT
              Pi USB  ──► camera
```

Never put motor voltage on GPIO. HAT does not power the Pi.

---

## CAD fit check

- Wheelbase **250 mm** (5″ prop clearance)  
- Motor pads **16×16 M3**  
- Stack holes **30.5×30.5**  
- Pi holes **58×49**  
- Battery under the deck; camera upright / forward-facing  

See [`cad/README.md`](../cad/README.md).
