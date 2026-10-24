"""
DermaScan – Full Training Pipeline
====================================
Fine-tunes EfficientNetB0 on the ISIC 9-class skin cancer dataset.

HOW TO RUN:
  1. Place your kaggle.json at:  /Users/aryan/Desktop/Neural/kaggle.json
     (Download it from https://www.kaggle.com/settings → "Create New Token")
  2. Run:  .venv/bin/python train_model.py
  3. After training, the best model is saved automatically as:
       skin_cancer_model.pkl  (used by the Flask backend)
       skin_cancer_model_trained.h5

WHAT THIS DOES:
  • Downloads the ISIC dataset from Kaggle (~2239 train images, 9 classes)
  • Applies heavy data augmentation (rotation, flip, zoom, brightness, contrast)
  • Phase 1 – trains only the new classification head (5 epochs, fast)
  • Phase 2 – unfreezes top-40 layers of EfficientNetB0 for fine-tuning (20 epochs)
  • Uses cosine-decay learning rate, early stopping, and best-weights checkpoint
  • Expected accuracy: 75–90% on validation set

Expected training time on Apple M-series chip: ~15–30 minutes total.
"""

import os
import sys
import pickle
import zipfile
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
)

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
KAGGLE_JSON     = os.path.join(BASE_DIR, "kaggle.json")
DATASET_ZIP     = os.path.join(BASE_DIR, "skin-cancer9-classesisic.zip")
DATA_DIR        = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR       = os.path.join(DATA_DIR, "Skin cancer ISIC The International Skin Imaging Collaboration", "Train")
TEST_DIR        = os.path.join(DATA_DIR, "Skin cancer ISIC The International Skin Imaging Collaboration", "Test")

PKL_PATH        = os.path.join(BASE_DIR, "skin_cancer_model.pkl")
H5_PATH         = os.path.join(BASE_DIR, "skin_cancer_model_trained.h5")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "best_checkpoint.keras")

IMAGE_SIZE      = (224, 224)
BATCH_SIZE      = 32
NUM_CLASSES     = 9

CLASS_NAMES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion",
]

CLASS_DESCRIPTIONS = {
    "actinic keratosis":         "Rough, scaly patch caused by sun damage. Pre-cancerous.",
    "basal cell carcinoma":      "Most common skin cancer. Slow-growing, rarely spreads.",
    "dermatofibroma":            "Benign fibrous nodule, usually on limbs.",
    "melanoma":                  "Dangerous skin cancer arising from pigment cells. Early detection critical.",
    "nevus":                     "Common mole. Usually benign but monitor for changes.",
    "pigmented benign keratosis": "Non-cancerous warty growth, harmless.",
    "seborrheic keratosis":      "Benign waxy growth. Common with age, not cancerous.",
    "squamous cell carcinoma":   "Second most common skin cancer. Can spread if untreated.",
    "vascular lesion":           "Abnormal blood vessel growth. Usually benign.",
}
# ─────────────────────────────────────────────────────────────────────────────


# ── Step 1 – Download dataset from Kaggle ─────────────────────────────────────
def download_dataset():
    if os.path.isdir(TRAIN_DIR):
        print("[INFO] Dataset already extracted. Skipping download.")
        return

    # Configure kaggle credentials
    if not os.path.exists(KAGGLE_JSON):
        print("\n" + "="*60)
        print("  ERROR: kaggle.json not found!")
        print("="*60)
        print(f"\n  Place your kaggle.json at:\n  {KAGGLE_JSON}")
        print("\n  How to get kaggle.json:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section → click 'Create New Token'")
        print("  3. Copy the downloaded kaggle.json to this folder")
        sys.exit(1)

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    dest = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(dest):
        shutil.copy(KAGGLE_JSON, dest)
        os.chmod(dest, 0o600)

    # Download via Kaggle API
    print("[INFO] Downloading ISIC skin cancer dataset from Kaggle …")
    import subprocess
    result = subprocess.run(
        [".venv/bin/python", "-m", "kaggle", "datasets", "download",
         "-d", "nodoubttome/skin-cancer9-classesisic",
         "-p", BASE_DIR],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        sys.exit(1)

    # Extract
    zip_path = os.path.join(BASE_DIR, "skin-cancer9-classesisic.zip")
    if os.path.exists(zip_path):
        print("[INFO] Extracting dataset …")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        print("[INFO] Extraction complete.")
    else:
        print("[ERROR] ZIP file not found after download.")
        sys.exit(1)


# ── Step 2 – Build data pipelines with augmentation ──────────────────────────
def build_datasets():
    # Strong augmentation to fight overfitting on a small dataset
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.3),
    ], name="augmentation")

    # Preprocessing for EfficientNetB0 (scales to [0,255] — model handles internally)
    def preprocess_train(image, label):
        image = tf.cast(image, tf.float32)
        image = augmentation(image, training=True)
        return image, label

    def preprocess_val(image, label):
        image = tf.cast(image, tf.float32)
        return image, label

    # Load train dataset (80/20 split)
    train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        label_mode="categorical",
        seed=42,
        subset="training",
        validation_split=0.2,
    )

    val_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        label_mode="categorical",
        seed=42,
        subset="validation",
        validation_split=0.2,
    )

    detected_classes = train_ds_raw.class_names
    print(f"[INFO] Detected classes: {detected_classes}")

    # Apply preprocessing + prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (train_ds_raw
                .map(preprocess_train, num_parallel_calls=AUTOTUNE)
                .cache()
                .shuffle(1000)
                .prefetch(AUTOTUNE))

    val_ds = (val_ds_raw
              .map(preprocess_val, num_parallel_calls=AUTOTUNE)
              .cache()
              .prefetch(AUTOTUNE))

    # Class weights to handle imbalance
    class_counts = {}
    for _, labels in train_ds_raw.unbatch():
        idx = int(tf.argmax(labels).numpy())
        class_counts[idx] = class_counts.get(idx, 0) + 1

    total = sum(class_counts.values())
    n_cls = len(class_counts)
    class_weights = {
        cls: total / (n_cls * count)
        for cls, count in class_counts.items()
    }
    print(f"[INFO] Class weights: { {k: round(v,2) for k,v in class_weights.items()} }")

    return train_ds, val_ds, class_weights, detected_classes


# ── Step 3 – Build model ───────────────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES) -> Model:
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling=None,
    )
    backbone.trainable = False   # Frozen for Phase 1

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image_input")
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dense(512, activation="relu", name="dense_1",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_2",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="dermascan_finetuned")
    return model


# ── Step 4 – Training ──────────────────────────────────────────────────────────
def train(model, train_ds, val_ds, class_weights):
    callbacks_common = [
        EarlyStopping(
            monitor="val_accuracy", patience=6,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            CHECKPOINT_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3,
            min_lr=1e-7, verbose=1
        ),
    ]

    # ── Phase 1: train head only ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("  PHASE 1 – Training classification head (frozen backbone)")
    print("="*55)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        class_weight=class_weights,
        callbacks=callbacks_common,
    )

    # ── Phase 2: unfreeze top-60 layers for fine-tuning ──────────────────────
    print("\n" + "="*55)
    print("  PHASE 2 – Fine-tuning top layers of EfficientNetB0")
    print("="*55)

    backbone = model.layers[1]   # EfficientNetB0 is at index 1
    backbone.trainable = True

    # Freeze all but the top 60 layers
    for layer in backbone.layers[:-60]:
        layer.trainable = False

    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    print(f"[INFO] {trainable_count} backbone layers unfrozen for fine-tuning")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),   # lower LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        class_weight=class_weights,
        callbacks=callbacks_common,
    )

    return history1, history2


# ── Step 5 – Save .pkl ────────────────────────────────────────────────────────
def save_pkl(model, class_names_detected):
    # Use notebook class names if detected ones match length
    final_classes = class_names_detected if len(class_names_detected) == NUM_CLASSES else CLASS_NAMES

    model.save(H5_PATH)
    print(f"\n[OK] Keras model saved → {H5_PATH}")

    payload = {
        "model":              model,
        "class_names":        final_classes,
        "class_descriptions": CLASS_DESCRIPTIONS,
        "image_size":         IMAGE_SIZE,     # (224, 224)
        "num_classes":        NUM_CLASSES,
        "temperature":        1.0,            # real trained model → no sharpening needed
    }

    with open(PKL_PATH, "wb") as fh:
        pickle.dump(payload, fh)

    size_mb = os.path.getsize(PKL_PATH) / 1024 / 1024
    print(f"[OK] Pickle file saved → {PKL_PATH}  ({size_mb:.1f} MB)")


# ── Step 6 – Evaluate ─────────────────────────────────────────────────────────
def evaluate(model, val_ds):
    print("\n[INFO] Evaluating on validation set …")
    loss, acc = model.evaluate(val_ds, verbose=1)
    print(f"\n{'='*55}")
    print(f"  Final Validation Accuracy : {acc*100:.2f}%")
    print(f"  Final Validation Loss     : {loss:.4f}")
    print(f"{'='*55}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  DermaScan – Full Training Pipeline")
    print("="*60)
    print(f"  TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPUs available: {len(gpus)}")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)

    # 1. Download dataset
    download_dataset()

    # 2. Build data pipelines
    train_ds, val_ds, class_weights, detected_classes = build_datasets()

    # 3. Build model
    model = build_model(NUM_CLASSES)
    model.summary()

    # 4. Train
    train(model, train_ds, val_ds, class_weights)

    # 5. Evaluate
    evaluate(model, val_ds)

    # 6. Save
    save_pkl(model, detected_classes)

    print("\n[DONE] Training complete! Restart the Flask backend to load the new model:")
    print("       pkill -f 'app.py' && .venv/bin/python app.py &")


if __name__ == "__main__":
    main()
