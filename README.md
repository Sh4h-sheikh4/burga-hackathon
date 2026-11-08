# Fire-drone: IR camera → TigerData → fire assessment → TigerData

> **What's in here.** Your zip (`Hackathon_Fall26` KiCad 10 project + `train_predict.py`) was reviewed. Findings are in
> **`docs/INSPECTION_NOTES.md`**. Short version:
> * **Model:** the original is a LinearSVC on raw pixels with fire/no_fire labels only. It has no severity, no confidence,
>   isn't a CNN, and won't train on fire-only data. It's preserved in `originals/user_upload/`. `model/train_predict.py`
>   is a revised CNN trainer for presence + severity that exports straight into this pipeline. **No weights or images
>   were supplied, so nothing is trained on real data and no accuracy is claimed.**
> * **PCB:** KiCad 10.0.8 ERC/DRC show 0 errors and 0 unconnected pads, with schematic parity OK. But the IMU's **SDx/SCx pins
>   (U2-2/U2-3) are floating, against the ST datasheet**, and **U2 has no local decoupling**. The IMU is wired for **SPI**,
>   so the pipeline now reads it over SPI.
> * The camera is described only as a "USB IR camera": near-IR vs thermal is unknown.

> **No Raspberry Pi? Start with [`HACKATHON_LAPTOP.md`](HACKATHON_LAPTOP.md)**: folder upload → TigerData → model → results.

## Architecture

```
 Raspberry Pi 3 (drone)                       TigerData service (PostgreSQL + TimescaleDB)         Ground server / laptop
 ┌─────────────────────────┐   INSERT frame   ┌───────────────────────────────┐   claim (SKIP     ┌─────────────────────┐
 │ USB IR camera (UVC)     │ ───────────────▶ │ frames (hypertable)           │ ◀──── LOCKED) ─── │ firedrone.worker    │
 │ LSM6DS3TR-C over SPI0   │                  │   image bytea + IMU + status  │                   │  model backend      │
 │ firedrone.ingest        │  offline? spool  │ fire_assessments (hypertable) │ ◀── INSERT ────── │  (ONNX/TFLite/Keras │
 │  → ./spool, retries     │  to SD card      │ fire_severity_1min (cont.agg) │                   │   /TorchScript)     │
 └─────────────────────────┘                  │ fire_latest_by_device (view)  │                   └─────────────────────┘
                                              └───────────────────────────────┘
```

* **The Pi only captures and uploads.** Full TensorFlow on a Pi 3 (1 GB RAM, Cortex-A53) is constrained and
  not recommended: it depends on whether a compatible build exists for your OS/Python, and memory and latency are tight.
  The default is to run the worker on a laptop/server. If you need on-board inference, a converted **TFLite** (or ONNX)
  model with `backend: "tflite"` is usually lighter. Benchmark it on the Pi before relying on it.
* The `frames` table is the work queue. Workers claim rows with `FOR UPDATE SKIP LOCKED`, so several workers can run
  at once without processing a frame twice (tested with 4 concurrent workers). If a worker crashes, its claimed frames
  are picked up again after `worker_stale_claim_s`.
* The output type is `fire_assessment_v1`: one row per frame per model version, written in the same transaction
  that marks the frame `done`.

## How the output works (no invented answers)

| status | fire_present | severity_class | when |
|---|---|---|---|
| `ok` | TRUE/FALSE | label or NULL | P(fire) is outside the uncertainty band, the frame looks usable, the domains match |
| `uncertain` | **NULL** | NULL | P(fire) is between `no_fire_at_or_below` and `fire_present_at_or_above`, OR the frame is low quality (dark/saturated/blurry/fast rotation from the IMU), OR the declared training domain differs from the camera's (e.g. RGB vs IR) |
| `model_unavailable` | **NULL** | NULL | no valid model artifact configured or it failed to load (the default) |
| `error` | NULL | NULL | inference kept failing after `worker_max_attempts` |

* Severity is reported only when the model actually has a severity output **and** its top class is at least
  `min_severity_confidence` confident. A binary model gets `severity_class = NULL` with a note saying so.
* `validated` stays FALSE until you set `model.validated: true` after `tools/evaluate.py` passes on labeled held-out IR images.
* `proxy_hot_pixel_fraction`, `proxy_max_intensity` and `proxy_mean_intensity` are **image-brightness proxies**,
  stored separately. They are not temperature and not hazard. The exact camera model is unknown: it could be a near-IR
  camera (reflected light, often with IR illuminators) or a thermal camera. Many cameras also apply automatic gain/exposure,
  so brightness may not be comparable between frames. Treat these as relative image statistics unless your camera is
  confirmed radiometric and calibrated. (`proxy_hot_pixel_fraction` only means "fraction of very bright pixels".)
* `is_synthetic` is carried from the frame to every output, so test data can't be mistaken for field data.

## Setup

### 1. TigerData
1. Create a service in the TigerData (Tiger Cloud) console and copy its connection string (it uses `sslmode=require`).
2. `cp .env.example .env` and paste the connection string into `FIREDRONE_DB_URL`.
3. Create the schema: `psql "$FIREDRONE_DB_URL" -f sql/001_schema.sql -f sql/002_tracking.sql`
   (or run `python -m firedrone.worker --init-schema --once`). Re-running it is safe.

The schema uses the current hypertable syntax from the TigerData docs, `CREATE TABLE … WITH (tsdb.hypertable, …)`
(TimescaleDB ≥ 2.20; tested on 2.23.0). `frames` has columnstore off because rows hold images and get status updates.
`fire_assessments` uses columnstore with `segmentby = device_id`.
**Storage:** a 640×480 JPEG is about 30–80 KB, so 1 frame/s comes to roughly 100–300 MB per flight hour.
Increase `capture_interval_s` (fewer frames per second), lower `jpeg_quality`/resolution, or move images to object storage and keep a URL, if that's too much.

### 2. Raspberry Pi 3 (Raspberry Pi OS Bookworm)
```bash
sudo raspi-config nonint do_spi 0          # enable SPI: the Hackathon_Fall26 HAT wires the IMU to SPI0/CE0
sudo apt install python3-opencv python3-psycopg python3-spidev
ls /dev/spidev0.0                           # must exist
pip install -r requirements-pi.txt
cp config/example.json config/pi.json       # set device_id, camera_index; imu_interface "spi" (default) or "i2c"
python -m firedrone.ingest --config config/pi.json --frames 5     # quick test
sudo cp deploy/firedrone-ingest.service /etc/systemd/system/ && sudo systemctl enable --now firedrone-ingest
```
If the link drops, frames go to `spool_dir` (the SD card) and upload automatically once the connection is back.
`--flush-spool` uploads them by hand.

### 3. Worker (server)
```bash
pip install -r requirements-server.txt      # plus tensorflow/torch if your model needs it
cp config/example.json config/server.json   # fill the "model" block (below)
python -m firedrone.worker --config config/server.json
```

### 4. Training the revised model and connecting it (`model` block)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu      # on a laptop/server, not the Pi
# data/{train,val,test}/{no_fire,low,moderate,high}/*.jpg   (frames from the same camera type as the drone)
python model/train_predict.py train --data data --out artifacts --domain ir --epochs 30
#   add --arch mobilenet_v3_small --pretrained (needs torchvision) for transfer learning
python model/train_predict.py predict frame.jpg --config artifacts/model_config.json
python -m firedrone.worker --config artifacts/model_config.json         # same thresholds as `predict`
```
`artifacts/model_config.json` is a complete `model` block (`backend: onnx`, input spec, labels) with `"validated": false`.
Merge it into `config/server.json`, and set `validated` yourself only after reviewing `artifacts/metrics.json` on a real
held-out test split. The original LinearSVC (`.joblib`) is not loaded by the worker: it has no probabilities, so
uncertain cases can't be detected. The fields below can also be filled in by hand for other models:

* `backend`: `onnx` | `tflite` | `keras` (`.h5`/`.keras`) | `torchscript` | `none`
* `input`: size, channels, `NHWC`/`NCHW`, and `scale` (`0_1`, `minus1_1`, `imagenet`, `raw_0_255`). **These must match
  your training preprocessing exactly.**
* `output.type` (these are the contracts the worker supports today; other designs such as ordinal/regression severity
  outputs or calibrated physical estimates would need a small adapter added to `firedrone/model.py`):
  * `binary_sigmoid`: one value = P(fire). You get no severity.
  * `softmax_classes`: e.g. labels `["no_fire","low","moderate","high"]`, in your training class-index order (Keras
    `image_dataset_from_directory` sorts folder names alphabetically, so the order may be `high, low, moderate, no_fire`!).
  * `two_head`: a fire head plus a severity head.
* `training_domain`: `rgb`, `ir` or `unknown`. If it differs from the camera's `sensor_domain`, confident outputs are
  downgraded to `uncertain` by default. **"ir" alone isn't enough to prove a match:** near-IR and thermal (LWIR) cameras
  produce very different images. Even with `training_domain = sensor_domain = "ir"`, only trust the model after it has
  been validated on frames from the same camera type.

### 5. Test data (synthetic)
`python -m firedrone.synthetic --frames 20 --init-schema` inserts generated grey blobs (`is_synthetic = TRUE`,
device `SYNTHETIC-drone`). These are not fire images. They only exercise the pipeline.

### 6. Queries for tracking severity
```sql
SELECT * FROM fire_latest_by_device;                                   -- current status per drone
SELECT bucket, n_fire, n_uncertain, max_severity_index, all_validated
FROM fire_severity_1min WHERE mission_id = 'mission-42' ORDER BY bucket; -- severity over time
SELECT a.*, f.gyro_x, f.latitude FROM fire_assessments a
JOIN frames f USING (frame_id, captured_at) WHERE a.status = 'ok' AND a.fire_present;
```

## Tests
```bash
export FIREDRONE_TEST_DSN=postgresql://user@host:port/db   # a disposable DB with TimescaleDB; tests DROP the tables
python -m pytest -q
```
50 tests (last run: 50 passed): 38 run without a database. They cover decision logic (incl. the uniform-output guard),
output mapping, preprocessing, the IMU over fake I2C and SPI buses, the model-source inspector, the PCB reviewer (I2C/SPI
fixtures), the revised trainer on synthetic images (training, ONNX-vs-PyTorch parity, refusing fire-only or too-small
data, predict CLI), and the user's original script run unchanged. 12 more run against PostgreSQL 17 + TimescaleDB 2.23:
schema, no-model path, ONNX end to end + continuous aggregate, 4 concurrent workers, dedupe, retry→error, offline
spool→flush, the worker using a trainer-produced model, and the laptop flow (folder upload incl. dedupe/mtime/pacing,
CLI upload→worker→results, requeue). Without `FIREDRONE_TEST_DSN`: 38 passed, 12 skipped.
None of this tests real fire images or the physical board.
`tools/make_test_onnx.py` builds a hand-weighted TEST-ONLY ONNX graph to exercise the plumbing. It is not a fire model.

## Tools
* `tools/triage_attachments.py FILES --copy-to originals/` identifies `.bin` files (h5/keras/pt/onnx/tflite/KiCad/zip). Originals are never modified.
* `tools/inspect_model_source.py model.py` is a static review: final layer, loss, whether a severity output exists, and common bugs.
* `tools/evaluate.py --config … --data heldout/` gives a confusion matrix, fire recall, false-alarm rate and uncertain rate.
* `tools/kicad_check.sh project_dir [out]` runs `kicad-cli` ERC + DRC (`--schematic-parity --refill-zones`) and keeps the raw
  reports; it uses the Flathub KiCad 10 if there's no native `kicad-cli`. Then it runs:
* `tools/pcb_review.py board.kicad_pcb`, which checks the Pi header and LSM6DS3TR-C pins against the datasheet for both I2C and SPI wiring (supply, GND, CS, SA0, floating SDx/SCx, bus pins, 100 nF
  decoupling incl. distance from the supply pins, stackup).
