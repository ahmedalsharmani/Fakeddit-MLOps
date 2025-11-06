import tensorflow as tf, pandas as pd, mlflow, mlflow.keras
from dataset_tfdata import make_ds
from model_textcnn_resnet import build_model
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.setdefaultencoding = lambda *args, **kwargs: None  # safety placeholder

# =========================================
# Experiment setup
# =========================================
mlflow.set_experiment("FakeNews_Fusion_Model")

train_csv = "data/train.csv"
val_csv   = "data/val.csv"

BATCH = 32
EPOCHS_HEAD = 3
EPOCHS_FT = 2

with mlflow.start_run(run_name="fusion_baseline"):
    # ---------- 1. Build and prepare model ----------
    model, text_vec = build_model(num_classes=2)

    texts = pd.read_csv(train_csv)["text"].astype(str).tolist()
    text_ds = tf.data.Dataset.from_tensor_slices(texts).batch(256)
    text_vec.adapt(text_ds)

    train_ds = make_ds(train_csv, batch=BATCH, shuffle=True)
    val_ds   = make_ds(val_csv,   batch=BATCH)

    mlflow.log_param("batch_size", BATCH)
    mlflow.log_param("epochs_head", EPOCHS_HEAD)
    mlflow.log_param("epochs_ft", EPOCHS_FT)

    # ---------- 2. Callbacks ----------
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.5, verbose=1),
    ]

    # ---------- 3. Stage 1 — Train heads ----------
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=cbs)
    mlflow.log_metric("val_acc_stage1", history.history["val_accuracy"][-1])

    # ---------- 4. Stage 2 — Fine-tune backbone ----------
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    unfrozen = 0
    for layer in reversed(model.layers):
        name = layer.name.lower()
        if any(k in name for k in ["conv", "block"]) and "conv1d" not in name:
            layer.trainable = True
            unfrozen += 1
            if unfrozen >= 40:
                break

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, callbacks=cbs)
    mlflow.log_metric("val_acc_stage2", history2.history["val_accuracy"][-1])

    # ---------- 5. Save and log model ----------
    model.save_weights("fusion_model.weights.h5")

    # Save full model in HDF5 format (Windows-safe)
    model.save("fusion_model_full.h5", save_format="h5")

    # Log artifacts to MLflow
    mlflow.log_artifact("fusion_model_full.h5")
    mlflow.log_artifact("fusion_model.weights.h5")

print("✅ Fusion model training tracked successfully in MLflow!")
