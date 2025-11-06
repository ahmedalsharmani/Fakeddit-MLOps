#build_csv.py
import argparse
from pathlib import Path
import pandas as pd

def pick_text_column(df):
    for c in ["title","clean_title","text","headline"]:
        if c in df.columns: return c
    raise KeyError("No text column found (looked for title/clean_title/text/headline)")

def pick_label_column(df):
    for c in ["2_way_label","label","binary_label"]:
        if c in df.columns: return c
    raise KeyError("No binary label column found (looked for 2_way_label/label/binary_label)")

def main(tsv, images_dir, out_csv):
    images_dir = Path(images_dir)
    df = pd.read_csv(tsv, sep="\t")

    # choose columns
    text_col  = pick_text_column(df)
    label_col = pick_label_column(df)

    # id -> image_path (we saved images as <id>.jpg)
    df["id"] = df["id"].astype(str)
    df["image_path"] = df["id"].apply(lambda x: str((images_dir / f"{x}.jpg").as_posix()))

    # keep only rows whose image file exists
    df = df[df["image_path"].apply(lambda p: Path(p).exists())]

    out = df[["id", text_col, label_col, "image_path"]].rename(
        columns={text_col: "text", label_col: "label"}
    )
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out):,} rows -> {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    main(args.tsv, args.images_dir, args.out_csv)
