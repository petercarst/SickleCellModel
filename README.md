# SickleVision

**AI-Powered Sickle Cell Detection System** — research/educational use only, not a certified medical device.

SickleVision is a full-stack web application that uses a fine-tuned DenseNet121 model to classify red blood cell blood-smear images as **Sickle Cell** or **Normal**. It also rejects images that aren't blood smears at all (photos, logos, documents, etc.) before they ever reach the classifier.

## 1. System Architecture

```
Browser
  → Laravel :8001            (UI, MySQL scan history)
    → FastAPI :8010           (preprocessing, out-of-distribution gate)
      → TensorFlow Serving :8501   (DenseNet121 model, Docker)
```

| Layer | Technology | Responsibility |
|---|---|---|
| Frontend / Web app | PHP Laravel 12, Blade, Bootstrap 5 | Upload UI, scan history, stats |
| Database | MySQL | Logs every scan (filename, class, confidence, timestamp) — images themselves are never stored |
| Prediction API | Python FastAPI | Image preprocessing (224×224, normalize), out-of-distribution rejection, calls TF Serving |
| Model server | TensorFlow Serving (Docker) | Runs the CNN, returns a sigmoid classification score plus a feature embedding |

## 2. Project Structure

```
SickleCellClassification/
├── sickleVision/            Laravel web app (the UI)
│   ├── app/Http/Controllers/PredictionController.php
│   ├── app/Models/Prediction.php
│   ├── database/migrations/ …create_predictions_table.php
│   ├── resources/views/     Blade templates (layouts/app, predictions/*)
│   └── .env                 DB + FASTAPI_URL config (never commit)
├── api/
│   ├── tf_serving.py        FastAPI bridge: preprocessing + OOD gate + TF Serving call
│   └── ood_calibration.json Calibrated out-of-distribution detector weights
├── models/sickle-cell/<N>/  Versioned TensorFlow SavedModel (Git LFS), served by TF Serving
├── dataset/{train,val,test}/{Negative,Positive}/  Split image dataset
├── training.ipynb           Model training notebook (documentation/exploration)
├── train.py                 Runnable training script (mirrors the notebook)
├── export_ood.py            Builds the dual-output model + out-of-distribution detector
└── requirements.txt         Python dependencies
```

## 3. Prerequisites

- PHP 8.2+ and [Composer](https://getcomposer.org)
- MySQL (e.g. via XAMPP)
- Python 3.10+ with the packages in `requirements.txt`
- Docker (to run TensorFlow Serving)

## 4. Setup

```bash
# 1. Python environment
pip install -r requirements.txt

# 2. Laravel app
cd sickleVision
composer install
cp .env.example .env
php artisan key:generate

# 3. Create the database, then migrate
#    (set DB_DATABASE/DB_USERNAME/DB_PASSWORD in .env to match your MySQL setup)
php artisan migrate
```

`sickleVision/.env` also sets `FASTAPI_URL` (default `http://127.0.0.1:8010/predict`) — the address Laravel calls for predictions.

## 5. Running the Application

**Quick start (Windows):** double-click `start-sicklevision.bat` — it brings up MySQL, Docker/TF Serving, the FastAPI bridge, and the Laravel dev server together (safe to re-run; only starts what isn't already running). `stop-sicklevision.bat` shuts down FastAPI, Laravel, and the TF Serving container (leaves MySQL and Docker Desktop running for other local projects). Logs land in `logs/`.

Or start each service manually, in its own terminal:

```bash
# Terminal 1 — TensorFlow Serving (serves the highest-numbered model version automatically)
docker run -d --name sickle-tfserving -p 8501:8501 \
  -v "<absolute-path-to-repo>/models/sickle-cell:/models/sickle-cell" \
  -e MODEL_NAME=sickle-cell tensorflow/serving

# Terminal 2 — FastAPI bridge
python api/tf_serving.py

# Terminal 3 — Laravel
cd sickleVision
php artisan serve --port=8001
```

Open **http://127.0.0.1:8001**.

## 6. How It Works

1. User drags/selects a blood smear image (JPEG, JPG, PNG, WebP, max 10 MB) on the Laravel Scan page.
2. Browser `POST`s to Laravel's `/predict` route.
3. Laravel forwards the file to the FastAPI bridge.
4. FastAPI resizes to 224×224, normalizes, and calls TF Serving, which returns both a classification score and a 1024-dim feature embedding.
5. FastAPI runs the **out-of-distribution gate**: a logistic-regression discriminator (trained on real blood smears vs. procedurally generated non-medical images) checks whether the image actually looks like a blood smear. If not, FastAPI returns **HTTP 422** and no classification is made.
6. Otherwise: score ≥ 0.5 → **Sickle Cell**, score < 0.5 → **Normal**.
7. Laravel logs the result (class, confidence, raw score, filename, timestamp — never the image itself) to MySQL and returns JSON to the browser, which renders the badge, confidence bar, and stats.

## 7. API Reference

### `POST /predict` (Laravel, and equivalently the FastAPI bridge directly)

| Field | Type | Description |
|---|---|---|
| `file` | multipart/form-data | Blood smear image (JPEG/JPG/PNG/WebP, max 10 MB) |

**Success (200):**
```json
{ "class": "Sickle Cell", "confidence": 0.9241, "raw_score": 0.9241 }
```

**Rejected — not a blood smear (422):**
```json
{ "error": "This doesn't look like a blood smear microscopy image. Please upload a Giemsa/Wright-stained peripheral blood smear." }
```

**Other errors (400/502/503):**
```json
{ "error": "Human-readable message" }
```

## 8. Model Training

`train.py` (mirrored in `training.ipynb`) trains the classifier in two phases — a frozen DenseNet121 head, then fine-tuning the top layers — with class weighting, dropout, early stopping, and evaluation on a true held-out test split. It exports a versioned TensorFlow SavedModel to `models/sickle-cell/<N>/`.

`export_ood.py` then rebuilds the trained model with a second output (the feature embedding), calibrates the out-of-distribution detector against real training images and synthetic non-medical images, and re-exports the dual-output serving model plus `api/ood_calibration.json`. Run this after any retrain:

```bash
python train.py
python export_ood.py
docker restart sickle-tfserving   # pick up the new model version
```

## 9. Environment Variables (`sickleVision/.env`)

| Variable | Default | Description |
|---|---|---|
| `DB_CONNECTION` / `DB_DATABASE` / `DB_USERNAME` / `DB_PASSWORD` | mysql / sickle_cell_classification / root / *(empty)* | MySQL connection |
| `FASTAPI_URL` | `http://127.0.0.1:8010/predict` | Address of the FastAPI bridge |
| `APP_URL` | `http://127.0.0.1:8001` | Base URL of the Laravel app |

`TF_SERVING_URL` (env var, defaults to `http://127.0.0.1:8501/v1/models/sickle-cell:predict`) configures where `api/tf_serving.py` looks for TensorFlow Serving.

## 10. Troubleshooting

| Problem | Solution |
|---|---|
| `Could not reach the prediction service` | FastAPI (`api/tf_serving.py`) isn't running — start it first |
| `TF Serving error` / 502 | TensorFlow Serving isn't running or the model failed to load — check `docker logs sickle-tfserving` |
| Every image gets rejected as "not a blood smear" | Re-run `export_ood.py` after retraining — the calibration must match the currently deployed model |
| `SQLSTATE[HY000] [2002]` on migrate | MySQL isn't running, or `.env` DB credentials are wrong |
| Port already in use | Another process is bound to 8001/8010/8501 — change the port in the relevant command/`.env` |

## 11. Security & Privacy Notes

- Never commit `sickleVision/.env` (contains DB credentials)
- Uploaded images are processed in memory and are **never written to disk or the database** — only the prediction result is logged
- File type and size are validated both client-side and server-side
- This application is for research/educational use only — not a certified medical device, and not a substitute for professional diagnosis

## 12. License & Disclaimer

This project is provided for research and educational purposes only. It is not approved for clinical diagnosis or medical decision-making. Always consult a qualified healthcare professional for medical advice.
