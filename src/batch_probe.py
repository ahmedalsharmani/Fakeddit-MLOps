# src/batch_probe.py
import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# --- Project paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# --- Import your model builder ---
from src.model_textcnn_resnet import build_model  # noqa: E402


def _prep_image(path: Path):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return tf.expand_dims(img, 0)  # [1,224,224,3]


def _pred_bundle(model, text: str, img_path: Path):
    """Return (with_image_conf, text_only_conf, image_only_conf, label_id)."""
    txt = tf.constant([text])

    # with image
    x = {"image": _prep_image(img_path), "text": txt}
    p = model.predict(x, verbose=0)[0]
    conf_with = float(np.max(p))
    y_with = int(np.argmax(p))

    # text-only (blank image)
    x_text_only = {
        "image": tf.zeros((1, 224, 224, 3), dtype=tf.float32),
        "text": txt,
    }
    p_t = model.predict(x_text_only, verbose=0)[0]
    conf_text = float(np.max(p_t))

    # image-only (empty text)
    x_img_only = {"image": x["image"], "text": tf.constant([""])}
    p_i = model.predict(x_img_only, verbose=0)[0]
    conf_img = float(np.max(p_i))

    delta = abs(conf_with - conf_text)
    return conf_with, conf_text, conf_img, y_with, delta


def main(split: str, n: int, output: str):
    # --- Build + adapt vectorizer on train texts ---
    model, text_vec = build_model(num_classes=2)
    train_texts = pd.read_csv(DATA / "train.csv")["text"].astype(str).tolist()
    text_vec.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(256))
    model.load_weights(str(ROOT / "fusion_model.weights.h5"))

    # --- Load split and make absolute image paths ---
    df = pd.read_csv(DATA / f"{split}.csv").copy()
    df["image_path"] = (ROOT / df["image_path"]).astype(str)
    df = df.dropna(subset=["text", "image_path", "label"]).reset_index(drop=True)
    df = df[df["image_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    if len(df) == 0:
        print("No usable rows found in the CSV.")
        return

    # sample n rows (balanced if possible)
    if n and n < len(df):
        # try to stratify by label
        df_pos = df[df["label"] == 1].sample(min(n // 2, len(df[df["label"] == 1])), random_state=42)
        df_neg = df[df["label"] == 0].sample(min(n - len(df_pos), len(df[df["label"] == 0])), random_state=42)
        df = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # build a pool of candidate images for mismatching
    all_img_paths = df["image_path"].tolist()

    rows = []
    for i, r in df.iterrows():
        text = str(r["text"])
        lbl = int(r["label"])
        img_ok = Path(r["image_path"])

        # choose a different row's image as the "bad" image
        j = random.randrange(len(all_img_paths))
        # ensure mismatch (different index)
        if str(all_img_paths[j]) == str(img_ok) and len(all_img_paths) > 1:
            j = (j + 1) % len(all_img_paths)
        img_bad = Path(all_img_paths[j])

        # predictions
        ok_cw, ok_ct, ok_ci, ok_pred, ok_d = _pred_bundle(model, text, img_ok)
        bad_cw, bad_ct, bad_ci, bad_pred, bad_d = _pred_bundle(model, text, img_bad)

        rows.append({
            "text": text,
            "label": lbl,
            "img_ok": str(img_ok),
            "img_bad": str(img_bad),

            "ok_conf_with": round(ok_cw, 3),
            "ok_conf_text": round(ok_ct, 3),
            "ok_conf_img":  round(ok_ci, 3),
            "ok_delta":     round(ok_d, 3),
            "ok_pred": ok_pred,

            "bad_conf_with": round(bad_cw, 3),
            "bad_conf_text": round(bad_ct, 3),
            "bad_conf_img":  round(bad_ci, 3),
            "bad_delta":     round(bad_d, 3),
            "bad_pred": bad_pred,
        })

    out_df = pd.DataFrame(rows)
    out_path = ROOT / output
    out_df.to_csv(out_path, index=False)
    print(f"Saved per-example results -> {out_path}")

    # summary
    def mean(col): return float(np.mean(out_df[col].astype(float))) if len(out_df) else 0.0
    print("\n=== SUMMARY (means over sampled rows) ===")
    print(f"ok_conf_with : {mean('ok_conf_with'):.3f}")
    print(f"ok_conf_text : {mean('ok_conf_text'):.3f}")
    print(f"ok_conf_img  : {mean('ok_conf_img'):.3f}")
    print(f"ok_delta     : {mean('ok_delta'):.3f}")
    print(f"bad_conf_with: {mean('bad_conf_with'):.3f}")
    print(f"bad_conf_text: {mean('bad_conf_text'):.3f}")
    print(f"bad_conf_img : {mean('bad_conf_img'):.3f}")
    print(f"bad_delta    : {mean('bad_delta'):.3f}")

    print("\nExpect: bad_delta > ok_delta (mismatched images should reduce confidence more).")

    # show top-10 where mismatch hurt the most
    out_df["delta_gain"] = out_df["bad_delta"] - out_df["ok_delta"]
    top = out_df.sort_values("delta_gain", ascending=False).head(10)
    print("\n--- Top 10 strongest mismatches by (bad_delta - ok_delta) ---")
    print(top[["text", "img_ok", "img_bad", "ok_delta", "bad_delta"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="CSV split name without .csv (train/val/test)")
    ap.add_argument("--n", type=int, default=100, help="number of rows to sample (balanced if possible)")
    ap.add_argument("--output", default="probe_results.csv")
    args = ap.parse_args()
    main(args.split, args.n, args.output)
