import tensorflow as tf
import pandas as pd
import os
import sys

# --- Ensure current folder (src) is in Python path ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ✅ FIX IMPORTS — remove "src." prefix
from dataset_tfdata import make_ds
from model_textcnn_resnet import build_model



train_csv = "data/train.csv"
val_csv   = "data/val.csv"

# Build model & get the TextVectorization layer handle
model, text_vec = build_model(num_classes=2)

# Adapt vectorizer on training texts
texts = pd.read_csv(train_csv)["text"].astype(str).tolist()
text_ds = tf.data.Dataset.from_tensor_slices(texts).batch(256)
text_vec.adapt(text_ds)

# Datasets
train_ds = make_ds(train_csv, batch=32, shuffle=True)
val_ds   = make_ds(val_csv,   batch=32)

# Callbacks
cbs = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.5, verbose=1),
]

# Stage 1 — train heads (ResNet frozen)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=3, callbacks=cbs)

# Stage 2 (fine-tune tail of the ResNet trunk)
# Keep BN layers frozen (common practice for stable fine-tuning)
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Unfreeze last ~40 conv-like layers from the image trunk
unfrozen = 0
for layer in reversed(model.layers):
    name = layer.name.lower()
    if any(k in name for k in ["conv", "block"]) and "conv1d" not in name:
        # likely part of ResNet, not the text branch
        layer.trainable = True
        unfrozen += 1
        if unfrozen >= 40:
            break

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=2, callbacks=cbs)

# Save WEIGHTS (avoid Windows .keras zip encoding issues)
model.save_weights("fusion_model.weights.h5")
print("✅ Saved weights -> fusion_model.weights.h5")
