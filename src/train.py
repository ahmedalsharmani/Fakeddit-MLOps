import os
import sys
import tensorflow as tf
import pandas as pd

# --- 🧩 Ensure the script can find modules inside /src ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dataset_tfdata import make_ds
    from model_textcnn_resnet import build_model
except ModuleNotFoundError as e:
    print("⚠️ Import failed, check folder structure. Current path:", os.getcwd())
    print("Files in src:", os.listdir(os.path.dirname(os.path.abspath(__file__))))
    raise e

# --- 🧠 Dummy minimal training workflow (for CI/CD test) ---
def main():
    print("✅ Starting training pipeline...")

    # Create small fake dataset (CI safe, no large data)
    x_train = tf.random.normal((32, 224, 224, 3))
    y_train = tf.keras.utils.to_categorical(tf.random.uniform((32,), 0, 2, dtype=tf.int32), 2)

    model = build_model()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=1, batch_size=8)
    model.save("fusion_model_ci.keras")

    print("✅ Model training complete and saved successfully!")

if __name__ == "__main__":
    main()
