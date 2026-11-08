# Laptop-only setup: TigerData ↔ model, no Raspberry Pi

```
 image folder ──upload_images──▶ TigerData: frames (pending) ──worker + model──▶ TigerData: fire_assessments
 (your laptop)                    status/queue                                  presence, severity, confidence
```
A folder of images on your laptop stands in for the drone camera. Every uploaded row is marked
`capture_source = 'file_replay'`, so it's never mistaken for live camera data. The IMU fields stay empty.
Everything below was rehearsed end to end on a local PostgreSQL + TimescaleDB.
Point `FIREDRONE_DB_URL` at your own TigerData service to run against the cloud.

**Truth in advertising:** the plumbing is complete and tested. How good the fire/severity answers are depends entirely on
the images you train with. No real fire dataset was provided, so no real accuracy is claimed. Every output carries
`validated = false` until you check the test metrics.

---

## 0. Prerequisites
* Python **3.10–3.13** (`python3 --version`; on Windows `py --version`).
* A TigerData service, plus its connection string from the TigerData console. Open your service's connection details;
  the string usually looks like `postgres://tsdbadmin:<password>@<host>:<port>/tsdb?sslmode=require`.
* Run every command from the **repo root** (`burga-hackathon/`).

## 1. Create a virtual environment and install dependencies (CPU only)
macOS / Linux:
```bash
cd fire-drone
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-laptop.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu     # training only (~200 MB)
```
Windows (PowerShell):
```powershell
cd fire-drone
py -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass      # only if Activate.ps1 is blocked
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-laptop.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
(On macOS the CPU index also works; plain `pip install torch` is fine too.)

## 2. Give the tools your connection string (never write it into a file)
macOS / Linux (input is hidden; paste and press Enter):
```bash
read -rs FIREDRONE_DB_URL && export FIREDRONE_DB_URL
```
Windows PowerShell (hidden input):
```powershell
$env:FIREDRONE_DB_URL = [System.Net.NetworkCredential]::new('', (Read-Host 'TigerData connection string' -AsSecureString)).Password
```
This lasts for the current terminal only. Every terminal you use below needs it set. Don't commit it, paste it into chat,
or put it in `config/*.json` (the code ignores a `db_url` key in config files anyway).

## 3. Connect and create the tables
```bash
python -m firedrone.init_db
```
Expected: `connected: PostgreSQL …, timescaledb 2.x` and then `schema ready; hypertables: fire_assessments, frames`.
It's safe to re-run. If the TimescaleDB version shown is older than 2.20, it stops with a message.

## 4. Optional, 2 minutes: smoke test before you have real data
```bash
python tools/make_smoke_dataset.py              # SYNTHETIC grey blobs -> smoke_data/ and demo_images/
python model/train_predict.py train --data smoke_data --out demo_artifacts --domain rgb --epochs 8 --size 64
python -m firedrone.upload_images demo_images --domain rgb --mission-id smoke-1
python -m firedrone.worker --config demo_artifacts/model_config.json --once
python -m firedrone.results --mission-id smoke-1
```
If you see `status ok` rows with `low/moderate/high`, the whole chain works. Those results mean nothing about fires.

## 5. Prepare the real dataset
```
data/
  train/ no_fire/  low/  moderate/  high/
  val/   no_fire/  low/  moderate/  high/
  test/  no_fire/  low/  moderate/  high/
```
* **Folder names must match exactly.** `no_fire` is required, because without negatives the model can't learn whether a fire is present.
* At least 30 training images per class (the trainer refuses fewer; `--min-per-class` overrides this for experiments only).
  A common split is about 70/15/15. Keep images from the same video or fire in the **same** split, otherwise test scores are inflated.
* Write down a one-line rule for low/moderate/high (e.g. visible flame area: small spot / one front / large or crown fire)
  and apply it consistently. These classes describe the image, not real-world danger.
* Laptop datasets are usually ordinary photos, so use `--domain rgb`. The drone's "USB IR camera" type is unknown (it
  could be near-IR or thermal). A real deployment needs training images from that same camera type; an RGB-trained model
  is only a demo.
* **No severity labels?** Use presence only: `data/{train,val,test}/{no_fire,fire}/` with `--classes no_fire,fire`.
  The pipeline then stores fire yes/no with `severity_class = NULL`.

## 6. Train
```bash
# 4-class presence + severity
python model/train_predict.py train --data data --out artifacts --domain rgb --epochs 30
# presence only
python model/train_predict.py train --data data --out artifacts --classes no_fire,fire --domain rgb --epochs 30
```
* **Time budget:** on CPU, the default small CNN at 128 px trains a few hundred images in minutes (the rehearsal:
  120 images, 64 px, 8 epochs in about 5 s). Under time pressure use `--epochs 10 --size 96`. For better accuracy with few
  images, use transfer learning: `pip install torchvision --index-url https://download.pytorch.org/whl/cpu`, then add
  `--arch mobilenet_v3_small --pretrained` (downloads ImageNet weights once; slower per epoch).
* Output in `artifacts/`:
  * `fire_model.onnx`: the exported model, with calibrated probabilities built in.
  * `model_config.json`: the complete config for the worker. The ONNX path inside it is relative, so moving the folder is fine.
  * `metrics.json`: confusion matrix, per-class precision/recall, fire recall/false-alarm rate and uncertain rate on `test/`.
* Check one image: `python model/train_predict.py predict path/to/img.jpg --config artifacts/model_config.json --domain rgb`

## 7. Upload images to TigerData (the "camera")
```bash
python -m firedrone.upload_images demo_images --domain rgb --mission-id demo-1
python -m firedrone.upload_images demo_images --domain rgb --mission-id demo-1 --interval 2   # simulate 1 frame / 2 s
```
* Accepts jpg/png/bmp/tif/webp, downsizes to 1280 px, stores JPEG plus the original file name. Unreadable files are reported and skipped.
* Re-running skips images already uploaded for that device. `--allow-duplicates` uploads them again (e.g. a second demo run).
* `--timestamps mtime` uses the files' modification times instead of upload time. `--device-id` defaults to `laptop-replay`.
* Use test images the model has **not** trained on.

## 8. Run the model on what's in the database
```bash
python -m firedrone.worker --config artifacts/model_config.json --once     # process everything pending, then exit
python -m firedrone.worker --config artifacts/model_config.json            # keep running (live demo; Ctrl+C to stop)
```
For a live demo, run the worker in one terminal and `upload_images … --interval 2` in a second one (set the env var there too).
After retraining, re-score old frames with `--requeue` (add `--mission-id demo-1` to limit it). Old outputs are kept,
because outputs are stored per model name and version (`--version` on train).

## 9. Look at the results
```bash
python -m firedrone.results --mission-id demo-1             # queue counts + latest outputs
python -m firedrone.results --mission-id demo-1 --notes     # with the reason for each uncertain/withheld result
python -m firedrone.results --save-image <frame_id> --out check.jpg   # pull an image back out of the database
```
The same in SQL (psql or any SQL editor connected to your service):
```sql
SELECT f.source_name, a.captured_at, a.status, a.fire_present, a.fire_probability,
       a.severity_class, a.severity_confidence, a.validated, a.notes
FROM fire_assessments a JOIN frames f USING (frame_id, captured_at)
WHERE a.mission_id = 'demo-1' ORDER BY a.captured_at;

SELECT capture_source, status, count(*) FROM frames GROUP BY 1, 2;      -- queue

CALL refresh_continuous_aggregate('fire_severity_1min', NULL, NULL);    -- severity over time (also refreshes every minute)
SELECT bucket, n_frames, n_fire, n_uncertain, max_severity_index FROM fire_severity_1min
WHERE mission_id = 'demo-1' ORDER BY bucket;
```

### What the output columns mean
| column | meaning |
|---|---|
| `status` | `ok`: confident. `uncertain`: P(fire) between 0.30 and 0.70, a nearly flat class distribution, a dark/blurry frame, or a training-vs-camera domain mismatch. `model_unavailable`: no model loaded. `error`: inference failed 3 times. |
| `fire_present` | TRUE / FALSE only when `status = ok`; otherwise NULL (unknown, never "no fire") |
| `fire_probability` | 1 − P(no_fire) |
| `severity_class` / `severity_index` | the most likely of low(0)/moderate(1)/high(2), only when a fire is confidently present **and** that class has ≥ 0.50 probability; NULL for presence-only models |
| `validated` | always FALSE from the trainer. Set `"validated": true` in `model_config.json` yourself only after reviewing `metrics.json` |
| `proxy_*` | image brightness statistics, not temperature, not hazard |

## Expected behaviour without a model
`python -m firedrone.worker --config config/example.json --once` stores every frame as `status = model_unavailable`,
`fire_present = NULL`. That's intended: no model, no answer.

## Troubleshooting
| symptom | fix |
|---|---|
| `FIREDRONE_DB_URL is not set` | redo step 2 in *this* terminal |
| connection timeout / refused | check the host/port in the string and that the service is running; some campus or office networks block outgoing database ports, so try a hotspot |
| `SSL` / authentication errors | keep `?sslmode=require`; re-copy the password from the console (special characters must stay URL-encoded) |
| `Cannot train: class 'x' has N training images` | add images or (experiments only) `--min-per-class N` |
| `--classes must include 'no_fire'` | create `no_fire/` folders with non-fire images |
| everything `uncertain` | run `--notes`. Typical causes: model uncertain (more/better data, `--arch mobilenet_v3_small --pretrained`), domain mismatch (`--domain` on train vs upload), or dark/blurry frames |
| worker says `processed 0 frames` | nothing pending: upload more, or use `--requeue` |
| upload says `0 uploaded, N skipped (duplicates)` | those images are already stored for that device (e.g. re-running step 4). Add `--allow-duplicates`, or use a new `--device-id` |
| `temperature hit its bound` warning while training | the model learned almost nothing on val; get more or cleaner data |
| `ModuleNotFoundError: torch` | step 1 torch install (training only; the worker doesn't need torch) |
| PowerShell won't run `Activate.ps1` | `Set-ExecutionPolicy -Scope Process Bypass` |
