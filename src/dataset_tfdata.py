from pathlib import Path
import tensorflow as tf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # project root

def _load_image(path):
    img = tf.io.read_file(path)
    # decode (jpeg/png/gif) and ensure static shape is known
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])  # <-- important: give it a known rank/shape
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img

def make_ds(csv_path, batch=32, shuffle=False):
    df = pd.read_csv(csv_path)
    # expected columns: id, text, label, image_path
    paths = (ROOT / df["image_path"]).astype(str).tolist()
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype("int64").tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, texts, labels))

    def _map(p, t, y):
        img = _load_image(p)
        return {"image": img, "text": t}, y

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(4096, reshuffle_each_iteration=True)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds
