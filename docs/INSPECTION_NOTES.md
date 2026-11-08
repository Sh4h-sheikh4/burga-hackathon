# Inspection notes: Hackathon_Fall26 HAT + `train_predict.py`

Inputs reviewed (from the user's zip, sha256 `23b89fc9…f3e`, extracted read-only to `originals/user_upload/`):

| file | bytes | sha256 (first 12) |
|---|---|---|
| Hackathon_Fall26.kicad_pro | 17811 | 13d931801cef |
| Hackathon_Fall26.kicad_sch | 110260 | 0fc7e4a5159b |
| Hackathon_Fall26.kicad_pcb | 108011 | c93079dd3b61 |
| train_predict.py | 2705 | 709038477e54 |

The five `.bin`/`.py` files from the first message never reached this machine. The zip replaces them.
**No trained model file (.joblib) and no images were supplied.** So nothing below measures real-world accuracy.

---

## 1. Model: `train_predict.py` (original). Verdict: it doesn't meet the requirement, so I wrote a revised version

### What the original actually is (code read + executed unchanged on synthetic images)
- It is **not a CNN**. It's a scikit-learn `LinearSVC` on raw 128×128 grayscale pixels flattened to 16,384 numbers
  (`descriptor()`, lines 21–26; `LinearSVC(class_weight="balanced")`, line 45).
- **Labels are hard-coded to `("fire", "no_fire")`** (line 32). It can only output a fire / no-fire label. **There is
  no severity output and no severity labels**, so it can't say how bad a fire is, whatever is done afterwards.
- **It can't train on fire-only data.** `load_dataset` exits with `No images found in data/train/no_fire` if the
  `no_fire` folder is empty (confirmed by running it: `docs/raw_reports/original_model_execution.txt`). If the 600 images
  are all fires, this script won't run at all. If it is given negatives, it can learn presence only.
- **No confidence:** `LinearSVC` has no `predict_proba` (confirmed on the saved model), and `predict` prints a single
  label. That makes it impossible to apply an "uncertain" threshold or track confidence over time.
- **Raw-pixel linear model:** no translation or scale invariance. It typically generalises poorly to new scenes, angles and
  altitudes. Grayscale loading drops colour, which is a loss if the training images are RGB but may be fine for
  single-channel IR frames.
- What works: the code is syntactically fine, trains, saves/loads with joblib, and reports validation accuracy when
  `data/val/{fire,no_fire}` exist. On synthetic blobs it trained and predicted fine. That's a code check, not accuracy.

### What I changed (the original is preserved unmodified)
`model/train_predict.py` is a revised drop-in with the same `train` / `predict` commands:
- A small **CNN** (PyTorch, 4 conv blocks, GroupNorm, global pooling). There's an optional `--arch mobilenet_v3_small
  --pretrained` for transfer learning (downloads ImageNet weights; usually much better with only hundreds of images).
- **Fire presence + severity in one softmax** over the classes you label (default `no_fire,low,moderate,high`).
  Presence = 1 − P(no_fire); severity = the most likely fire class. Use `--classes no_fire,fire` for a presence-only model;
  the pipeline then reports no severity. Other severity designs (ordinal losses, regression on a measured quantity,
  calibrated physical models) are valid alternatives. This one was chosen because it matches labelled image folders.
- **Refuses to train** without a `no_fire` class or with fewer than `--min-per-class` (default 30) images per class.
  It requires a `val/` split for early stopping, and uses `test/` for metrics.
- **Temperature-scaled probabilities** (fit on val, baked into the ONNX export), with a warning if calibration fails.
- Uses **the same preprocessing code as the inference worker** (`firedrone.imaging.preprocess`), so training and
  deployment can't drift apart.
- Writes `fire_model.onnx`, `fire_model.pt`, `metrics.json` (confusion matrix, per-class precision/recall, fire recall and
  false-alarm rate at the pipeline thresholds, uncertain rate, ordinal severity error) and **`model_config.json`, a ready
  `model` block for the TigerData worker, always with `"validated": false`.**

### Pipeline fixes found while integrating
- With 4 classes, a know-nothing (uniform) output gives P(fire) = 0.75, which would have passed the 0.70 threshold. The
  worker now marks near-uniform class distributions as `uncertain` (`max_normalized_entropy`, default 0.90).
- A 2-class `["no_fire","fire"]` softmax model no longer produces a fake severity class.

### What's still needed before outputs mean anything
1. **Severity labels:** a written rubric (e.g. visible flame-front extent or burning-area fraction per frame at a known
   altitude) and images sorted into `no_fire/low/moderate/high`. Note that any class judged from photos describes the
   image, not true danger: real hazard depends on wind, fuel, terrain, spread rate and proximity to people.
2. **Negatives:** `no_fire` images, including look-alikes (sun glare, sunsets, warm rocks/roads, vehicles, lights).
3. **Matching camera domain:** the camera is only described as a "USB IR camera module". It might be near-IR (reflected
   light) or thermal, and the two look very different. Train or fine-tune on frames from that same camera type, and set
   `--domain` honestly.
4. **Held-out test images** from different flights or fires, then review `metrics.json` before setting `validated: true`.
5. 600 images split over 4 classes is small. Expect to need more, plus transfer learning and augmentation.

---

## 2. PCB / schematic: Hackathon_Fall26 (Raspberry Pi HAT, KiCad 10.0)

### Tool runs (raw outputs in `docs/raw_reports/user_board/`)
KiCad **10.0.8** `kicad-cli` (Flathub build), on a scratch copy, with the stock KiCad 10 libraries configured:
- **ERC:** 0 errors, 124 warnings. 93 off-grid endpoints, 21 isolated labels (unused GPIO labels from the HAT template),
  8 dangling wire ends, 2 symbol-vs-library differences (C1 `C_Small`, JP1 `SolderJumper_2_Open`; cosmetic).
  Three ERC checks are disabled in the project settings ("Global label only appears once", "Four connection points joined",
  "Footprint doesn't match footprint filters").
- **DRC** (`--schematic-parity --all-track-errors --refill-zones`): **0 errors, 0 unconnected pads, 0 schematic-parity
  issues**. 5 warnings: 3 dangling track stubs, and 2 footprint-vs-library differences (U2, J1; benign, see below).
- That matches your "no ERC/DRC errors". **But ERC/DRC don't check datasheet requirements.** The findings below come from
  reading the netlist and layout against the ST datasheet.

### Board summary
Built on the KiCad Raspberry Pi HAT template: J1 2×20 socket on the bottom (B.Cu), U1 CAT24C256 ID EEPROM (A0–A2 = GND
→ 0x50; ID_SDA/ID_SCL with R1/R2 3.9 kΩ pull-ups to +3V3; JP1 WP jumper), U2 LSM6DS3 (LGA-14), C1/C5 100 nF, C6 10 µF,
4 × M2.5 holes. Two copper layers, 1.6 mm FR4, 35 µm copper; all tracks 0.2 mm; 19 vias 0.9/0.6 mm; no copper pour
(the only zone is the template's "PoE" rule area). Outline 65 × 56 mm, holes on a 58 × 49 mm pattern 3.5 mm from the
edges, with camera and display cable slots.

### IMU wiring: U2 is on SPI, not I2C (valid design)
| U2 pin | function (ST DS Table 2) | net | Pi header | status |
|---|---|---|---|---|
| 8 VDD | 1.71–3.6 V supply | +3V3 | 1, 17 | OK |
| 5 VDDIO | I/O supply | +3V3 | 1, 17 | OK (3.3 V logic matches the Pi) |
| 6, 7 GND | ground | GND | 6, 9, 14, 20, 25, 30, 34, 39 | OK |
| 12 CS | 0 = SPI | /GPIO8 SPI0.CE0 | 24 | OK |
| 13 SCL/SPC | SPI clock | /GPIO11 SPI0.SCLK | 23 | OK |
| 14 SDA/SDI | SPI data in | /GPIO10 SPI0.MOSI | 19 | OK |
| 1 SDO/SA0 | SPI data out | /GPIO9 SPI0.MISO | 21 | OK |
| 4 INT1 | interrupt | /GPIO17 | 11 | OK (optional data-ready) |
| 9 INT2 | interrupt | unconnected (NC flag) | n/a | OK (output, may float) |
| 10, 11 NC | leave unconnected | unconnected | n/a | OK |
| **2 SDx** | **Mode 1: connect to VDDIO or GND** | **unconnected (NC flag)** | n/a | **DEFECT** |
| **3 SCx** | **Mode 1: connect to VDDIO or GND** | **unconnected (NC flag)** | n/a | **DEFECT** |

Software consequence: the IMU **must be read over SPI** (`/dev/spidev0.0`). The original pipeline assumed I2C at 0x6A,
so it's now SPI by default (`imu_interface: "spi"`), and the driver sets `I2C_disable` (CTRL4_C bit 2), as recommended
for SPI use.

### Defects (actionable; I did not edit the board)
1. **U2 pins 2 (SDx) and 3 (SCx) are floating. This breaks a datasheet rule, and ERC can't see it.**
   Evidence: schematic `no_connect` flags on U2 pins 2, 3 and 9. In the PCB, pads 2/3 sit on nets
   `unconnected-(U2-SDX-Pad2)` and `unconnected-(U2-SCX-Pad3)` with no copper. ST LSM6DS3TR-C datasheet Table 2: "SDx/SCx:
   Connect to VDDIO or GND" (Mode 1), and AN5130 §3.6: "in connection Mode 1, the SCx/SDx pins cannot be left floating.
   It's recommended to connect both of them to VDDIO." Their internal pull-ups are off by default, so they are floating
   CMOS inputs. That risks extra supply current, noise sensitivity and undefined behaviour.
   **Fix:** delete the two no-connect flags and tie U2-2 and U2-3 to +3V3 (VDDIO); GND is also allowed. Short traces next
   to U2's VDDIO pad are enough. Re-run ERC/DRC. (On an assembled board, the LGA pads aren't reachable for a bodge wire.)
2. **No local decoupling at U2, and only one 100 nF for its two supply pins.**
   Evidence: the schematic places C5 100 nF and C6 10 µF on U2 (C1 100 nF sits at the EEPROM U1). On the PCB, U2 is at
   (134.5, 71.1) mm. The nearest capacitor pad to U2 pin 8 (VDD) is C1 at 15.4 mm, and C6 is at 19.3 mm. C5, the one drawn
   for U2, is 28.3 mm away near U1. The nearest to pin 5 (VDDIO) is C1 at 13.7 mm. The datasheet recommends a 100 nF filter
   capacitor on **each** of VDD and VDDIO (Table 2 note 1).
   **Fix:** add a second 100 nF. Place one 100 nF at pin 8 and one at pin 5, each within about 1–2 mm with a short GND
   return, and keep the 10 µF nearby. Then move C1 next to U1 pin 8 (it's 23.8 mm away now), as the schematic intends.
3. **Minor cleanup:** 3 dangling track stubs (DRC warnings): /ID_SDA 4.0 mm at (138.5, 95.5), /ID_SCL 0.19 mm at
   (143.0, 94.0), +3V3 1.05 mm at J1 pin 1 (108.37, 48.77). Delete them.

### Checked and OK, or notes
- **Power:** the HAT only uses +3V3 from the Pi (IMU around 1 mA, EEPROM µA–mA), so 0.2 mm tracks are fine. **Nothing
  connects to +5V** beyond J1 pins 2 and 4, so **the HAT does not power the Pi**. On the drone, the Pi still needs its own
  5 V supply (micro-USB). If you later want to feed 5 V through the HAT, the Pi HAT design guide's back-powering
  protection and a regulator rated for the Pi plus camera are needed. That would be a design addition.
- **Grounds:** all 8 Pi GND pins and U2 pins 6/7 are on one GND net, connected by tracks only (0 unconnected pads).
  A GND pour on B.Cu would lower ground impedance and help with motor/ESC noise on a drone. That's a recommendation, not a defect.
- **ID EEPROM:** this is the template's standard circuit. It needs flashing with a valid HAT EEPROM image, or the Pi
  firmware ignores it; it's harmless if left blank. With JP1 open, WP relies on the CAT24C256's internal pull-down
  (check this for the exact part you buy).
- **Footprints:** U2's pads are **identical** to the KiCad 10 library `LGA-14_3x2.5mm_P0.5mm_LayoutBorder3x4y`. The DRC
  "mismatch" is one extra silkscreen line and extra properties. J1's differences are the expected mirroring from being
  on B.Cu, plus oval vs circle pads of the same 1.7 mm size. I didn't independently check the LGA land pattern against
  ST's package drawing.
- **Part value:** the symbol and value are `LSM6DS3` (the original part). The pinout is the same as the LSM6DS3TR-C, but
  set the BOM value/MPN to **LSM6DS3TR-C** so the right part gets ordered. The firmware here checks WHO_AM_I = 0x6A
  (LSM6DS3TR-C).
- **JP1:** the symbol is a solder jumper, but the footprint is a 2-pin header, so it needs a header plus a shunt, not a solder bridge.
- **USB:** there's no USB on the HAT. The USB IR camera plugs into the Pi 3 directly. Check the camera's current draw
  against the Pi's USB budget and your 5 V supply.
- **Mechanical:** the outline measures 56.0 mm tall. Check that against the HAT outline you're targeting. The mounting
  pattern (58 × 49 mm) and header position match a standard HAT.

### Limits of this review
No physical board, so no measurements: solder quality of the LGA, supply noise, SPI signal integrity, vibration, EMI and
thermal behaviour are unverified. The fab's capabilities aren't known. ERC/DRC were run with stock KiCad 10.0.8 libraries,
not necessarily the library versions you used. Bring-up check on hardware: enable SPI, run
`python -m firedrone.ingest --frames 1` and confirm the IMU reports WHO_AM_I = 0x6A.

## 3. Tool and environment notes
- KiCad 10 isn't in Debian's archive (that only has 9.0.2, which can't open 10.x files). It was installed from Flathub
  (`org.kicad.KiCad` 10.0.8). `tools/kicad_check.sh` uses the native `kicad-cli` or falls back to the flatpak.
- Renders produced by kicad-cli are in `docs/renders/` (schematic, copper layers, 3D top/bottom). They're real exports,
  not edited.
- Database tests ran on a local PostgreSQL 17.11 with TimescaleDB 2.23.0, **not a TigerData cloud service** (no credentials).
