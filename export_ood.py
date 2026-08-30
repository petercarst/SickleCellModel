"""
Builds a dual-output serving model (embedding + classification) from the
already fine-tuned checkpoint, then calibrates an out-of-distribution (OOD)
gate that rejects images which don't look like a blood smear microscopy
image *before* trusting the classifier's verdict — the classifier itself was
only ever trained on two classes of blood smear and will happily (and
wrongly) score an unrelated photo as one of them.

Plain distance-to-centroid in embedding space turned out not to separate
real photos/logos from blood smears reliably (tested empirically). Instead
this trains a small logistic-regression discriminator on:
  - the 1024-dim DenseNet121 GAP embedding (general image content), plus
  - a handful of hand-engineered color/texture stats (staining color
    signature, texture density) that the fine-tuned embedding alone doesn't
    cleanly capture.
Positive examples are the real training images; negative examples are
procedurally generated non-medical images (noise, solid colors, gradients,
shapes, checkerboards, blurred color blobs, skin-tone patches, sky/ground
scenery, text) so no external dataset is needed.

Run: python export_ood.py
"""
import glob
import json
import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

IMAGE_SIZE = 224
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ── 1. Dual-output serving model (embedding + classification) ──────────────
trained = tf.keras.models.load_model("checkpoints/finetune_best.keras")
embedding_layer = trained.get_layer("global_average_pooling2d")
serving_model = Model(
    inputs=trained.input,
    outputs={"embedding": embedding_layer.output, "classification": trained.output},
)
embed_model = Model(trained.input, embedding_layer.output)


# ── 2. Synthetic "not a blood smear" image generator ────────────────────────
def random_negative_image(size=IMAGE_SIZE):
    kind = random.choice([
        "noise", "solid", "gradient", "shapes", "checker",
        "photo_like", "text", "skin", "scenery",
    ])
    if kind == "noise":
        img = Image.fromarray(np.random.randint(0, 256, (size, size, 3), dtype=np.uint8))
    elif kind == "solid":
        img = Image.new("RGB", (size, size), tuple(int(x) for x in np.random.randint(0, 256, 3)))
    elif kind == "gradient":
        c1, c2 = np.random.randint(0, 256, 3), np.random.randint(0, 256, 3)
        arr = np.tile(np.linspace(c1, c2, size).astype(np.uint8), (size, 1, 1)).transpose(1, 0, 2)
        img = Image.fromarray(arr)
    elif kind == "shapes":
        img = Image.new("RGB", (size, size), tuple(int(x) for x in np.random.randint(180, 256, 3)))
        d = ImageDraw.Draw(img)
        for _ in range(random.randint(3, 10)):
            c = tuple(int(x) for x in np.random.randint(0, 256, 3))
            x0, x1 = sorted(np.random.randint(0, size, 2).tolist())
            y0, y1 = sorted(np.random.randint(0, size, 2).tolist())
            x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
            shape = random.choice(["ellipse", "rectangle", "line"])
            if shape == "line":
                d.line([x0, y0, x1, y1], fill=c, width=3)
            else:
                getattr(d, shape)([x0, y0, x1, y1], fill=c, outline=c, width=3)
    elif kind == "checker":
        n = random.choice([4, 8, 16])
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        c1, c2 = np.random.randint(0, 256, 3), np.random.randint(0, 256, 3)
        step = size // n
        for i in range(n):
            for j in range(n):
                arr[i*step:(i+1)*step, j*step:(j+1)*step] = c1 if (i + j) % 2 == 0 else c2
        img = Image.fromarray(arr)
    elif kind == "photo_like":
        arr = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        img = Image.fromarray(arr).resize((size, size), Image.BICUBIC).filter(ImageFilter.GaussianBlur(8))
    elif kind == "text":
        img = Image.new("RGB", (size, size), (255, 255, 255))
        d = ImageDraw.Draw(img)
        for _ in range(20):
            x, y = np.random.randint(0, size, 2).tolist()
            d.text((x, y), random.choice("ABCDEFGHabcdefgh0123"), fill=(0, 0, 0))
    elif kind == "skin":
        # portrait-photo proxy: warm skin-tone blobs on a soft background
        base = np.array([np.random.randint(150, 235), np.random.randint(110, 190), np.random.randint(90, 170)])
        arr = np.tile(base, (size, size, 1)).astype(np.uint8)
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)
        for _ in range(random.randint(2, 5)):
            c = tuple(int(x) for x in np.clip(base + np.random.randint(-30, 30, 3), 0, 255))
            x0, y0 = np.random.randint(0, size, 2).tolist()
            r = np.random.randint(20, 80)
            d.ellipse([x0-r, y0-r, x0+r, y0+r], fill=c)
        img = img.filter(ImageFilter.GaussianBlur(6))
    elif kind == "scenery":
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        horizon = np.random.randint(size//3, 2*size//3)
        sky = np.random.randint(120, 200, 3); ground = np.random.randint(60, 160, 3)
        arr[:horizon] = sky
        arr[horizon:] = ground
        img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(4))
    return img


def engineered_features(pil_img):
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
    ], dtype=np.float32)


def full_features(pil_img):
    arr = np.array(pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
    emb = embed_model.predict(arr[None, ...], verbose=0)[0]
    return np.concatenate([emb, engineered_features(pil_img)])


# ── 3. Build the training set ───────────────────────────────────────────────
pos_files = glob.glob("dataset/*/*/*.jpg")
print(f"Extracting features for {len(pos_files)} real blood smear images...")
X_pos = np.array([full_features(Image.open(f)) for f in pos_files])

N_NEG = 320
print(f"Generating and extracting features for {N_NEG} synthetic non-smear images...")
X_neg = np.array([full_features(random_negative_image()) for _ in range(N_NEG)])

X = np.concatenate([X_pos, X_neg])
y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
X_scaled = (X - mean) / std

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=SEED)
clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test), target_names=["not-smear", "smear"]))

# Real training images should score far above the decision threshold, with
# a comfortable margin below it for anything that isn't a blood smear.
train_probs = clf.predict_proba(X_scaled[: len(X_pos)])[:, 1]
threshold = float(min(0.85, train_probs.min() - 0.1))
print(f"min P(smear) on real images: {train_probs.min():.4f} -> using threshold {threshold:.4f}")

os.makedirs("api", exist_ok=True)
with open("api/ood_calibration.json", "w") as f:
    json.dump({
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "clf_weights": clf.coef_[0].tolist(),
        "clf_bias": float(clf.intercept_[0]),
        "threshold": threshold,
    }, f)
print("Saved api/ood_calibration.json")

# ── 4. Export the dual-output SavedModel for TF Serving (next version) ─────
export_root = "models/sickle-cell"
os.makedirs(export_root, exist_ok=True)
existing_versions = [int(d) for d in os.listdir(export_root) if d.isdigit()]
next_version = max(existing_versions, default=0) + 1
export_path = f"{export_root}/{next_version}"
serving_model.export(export_path)
print(f"Exported dual-output serving model to {export_path}")
