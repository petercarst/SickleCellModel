"""
Standalone training script for the sickle-cell classifier.

Mirrors training.ipynb — a DenseNet121 transfer-learning model, trained in two
phases (frozen head, then fine-tuned) and exported as a versioned TensorFlow
SavedModel that TF Serving can pick up automatically.

Run: python train.py
"""
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)
tf.random.set_seed(42)

IMAGE_SIZE = 224
BATCH_SIZE = 32

# ── Data ──────────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.1,
    brightness_range=(0.8, 1.2),
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    "dataset/train", target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
    class_mode="binary", seed=42,
)
val_generator = val_datagen.flow_from_directory(
    "dataset/val", target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False, seed=42,
)
test_generator = test_datagen.flow_from_directory(
    "dataset/test", target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False, seed=42,
)

print("class_indices:", train_generator.class_indices)

class_labels = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight="balanced", classes=np.unique(class_labels), y=class_labels
)
class_weights = dict(enumerate(class_weights_array))
print("class_weights:", class_weights)

# ── Model ─────────────────────────────────────────────────────────────────
base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model.trainable = False

inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs, outputs)

METRICS = [
    "accuracy",
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS)
model.summary()

os.makedirs("checkpoints", exist_ok=True)

# ── Phase 1: train the head ─────────────────────────────────────────────
head_callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint("checkpoints/head_best.keras", monitor="val_loss", save_best_only=True),
]
history_head = model.fit(
    train_generator, validation_data=val_generator, epochs=30,
    class_weight=class_weights, callbacks=head_callbacks,
)

# ── Phase 2: fine-tune the top of DenseNet121 ────────────────────────────
FINE_TUNE_AT = len(base_model.layers) - 30
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=METRICS)

fine_tune_callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint("checkpoints/finetune_best.keras", monitor="val_loss", save_best_only=True),
]
history_finetune = model.fit(
    train_generator, validation_data=val_generator, epochs=20,
    class_weight=class_weights, callbacks=fine_tune_callbacks,
)

# ── Evaluate on the held-out test set ────────────────────────────────────
results = model.evaluate(test_generator, return_dict=True)
print("Test results:", results)

test_generator.reset()
steps = int(np.ceil(test_generator.samples / test_generator.batch_size))
y_true = test_generator.classes
y_prob = model.predict(test_generator, steps=steps).ravel()
y_pred = (y_prob > 0.5).astype(int)

idx_to_class = {v: k for k, v in test_generator.class_indices.items()}
target_names = [idx_to_class[i] for i in sorted(idx_to_class)]

print(classification_report(y_true, y_pred, target_names=target_names))
print("Confusion matrix (rows = actual, cols = predicted):")
print(confusion_matrix(y_true, y_pred))

# ── Export for TF Serving (auto-incrementing version) ────────────────────
export_root = "models/sickle-cell"
os.makedirs(export_root, exist_ok=True)
existing_versions = [int(d) for d in os.listdir(export_root) if d.isdigit()]
next_version = max(existing_versions, default=0) + 1
export_path = f"{export_root}/{next_version}"

model.export(export_path)
print(f"Saved model to {export_path}")
