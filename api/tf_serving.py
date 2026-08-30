import json
import os
from io import BytesIO

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="SickleVision TensorFlow API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TensorFlow Serving endpoint (override with TF_SERVING_URL for other hosts/ports)
TF_SERVING_URL = os.environ.get(
    "TF_SERVING_URL", "http://127.0.0.1:8501/v1/models/sickle-cell:predict"
)

# train_generator.class_indices == {'Negative': 0, 'Positive': 1}, so a sigmoid
# output near 1 means "Positive" (sickle cell), near 0 means "Negative" (normal).
CLASS_NAMES = {0: "Normal", 1: "Sickle Cell"}
IMAGE_SIZE = 224
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Out-of-distribution gate: the serving model (see export_ood.py) also returns
# a 1024-dim feature embedding alongside the classification score. That
# embedding, plus a few hand-engineered color/texture stats, feed a small
# logistic-regression discriminator (trained on real blood smears vs.
# procedurally generated non-medical images) that rejects anything that
# doesn't look like a blood smear *before* trusting the (meaningless, for
# such an image) classification score. See export_ood.py for how this was
# calibrated.
_OOD_PATH = os.path.join(os.path.dirname(__file__), "ood_calibration.json")
with open(_OOD_PATH) as f:
    _ood = json.load(f)
OOD_FEATURE_MEAN = np.array(_ood["feature_mean"], dtype=np.float64)
OOD_FEATURE_STD = np.array(_ood["feature_std"], dtype=np.float64)
OOD_CLF_WEIGHTS = np.array(_ood["clf_weights"], dtype=np.float64)
OOD_CLF_BIAS = float(_ood["clf_bias"])
OOD_THRESHOLD = float(_ood["threshold"])


def engineered_features(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((100, 100))
    hsv = np.array(img.convert("HSV")).astype(np.float32)
    h, s, v = hsv[..., 0] / 255 * 360, hsv[..., 1] / 255, hsv[..., 2] / 255
    colored_mask = (s > 0.08) & (v > 0.15) & (v < 0.97)
    colored_frac = colored_mask.mean()
    gray = np.array(img.convert("L")).astype(np.float32)
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.array([
        s.mean(), s.std(), v.mean(), v.std(),
        colored_frac, grad_mag.mean(), grad_mag.std(),
        h[colored_mask].std() if colored_mask.sum() > 5 else 0.0,
    ], dtype=np.float64)


def smear_probability(embedding: list, pil_img: Image.Image) -> float:
    features = np.concatenate([np.array(embedding, dtype=np.float64), engineered_features(pil_img)])
    scaled = (features - OOD_FEATURE_MEAN) / OOD_FEATURE_STD
    logit = scaled @ OOD_CLF_WEIGHTS + OOD_CLF_BIAS
    return float(1 / (1 + np.exp(-logit)))


def read_file_as_image(data: bytes) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(BytesIO(data)).convert("RGB")
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return image, np.array(resized).astype(np.float32) / 255.0


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                TF_SERVING_URL.replace(":predict", "")
            )
        return {"status": "ok", "tf_serving_reachable": response.status_code == 200}
    except httpx.RequestError:
        return {"status": "ok", "tf_serving_reachable": False}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPEG, JPG, PNG, or WebP image.",
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")

    try:
        pil_image, image = read_file_as_image(raw)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")

    img_batch = np.expand_dims(image, 0)
    json_data = {"instances": img_batch.tolist()}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(TF_SERVING_URL, json=json_data)
            response.raise_for_status()
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Could not reach TF Serving: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"TF Serving error: {e.response.text}")

    result = response.json()["predictions"][0]
    if smear_probability(result["embedding"], pil_image) < OOD_THRESHOLD:
        raise HTTPException(
            status_code=422,
            detail=(
                "This doesn't look like a blood smear microscopy image. "
                "Please upload a Giemsa/Wright-stained peripheral blood smear."
            ),
        )

    prediction = float(result["classification"][0])
    predicted_label = 1 if prediction >= 0.5 else 0
    confidence = prediction if predicted_label == 1 else 1 - prediction

    return {
        "class": CLASS_NAMES[predicted_label],
        "confidence": confidence,
        "raw_score": prediction,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
