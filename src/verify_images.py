import os
import pandas as pd
from pathlib import Path

# Define base paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMG_DIR = ROOT / "images"

# CSV files to check
csv_files = ["train.csv", "val.csv", "test.csv"]

missing = []

for csv_name in csv_files:
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        print(f"⚠️  Missing CSV file: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    total = len(df)

    # Build absolute image paths
    df["abs_path"] = df["image_path"].apply(lambda x: (ROOT / x).as_posix())

    # Check which ones are missing
    df["exists"] = df["abs_path"].apply(os.path.exists)
    missing_df = df[~df["exists"]]

    if len(missing_df) > 0:
        print(f"❌ {len(missing_df)} missing images in {csv_name}:")
        print(missing_df[["image_path"]].head(10))  # show only first few
        missing.extend(missing_df["image_path"].tolist())
    else:
        print(f"✅ All {total} images found for {csv_name}.")

print("\nSummary:")
if missing:
    print(f"Total missing images: {len(missing)}")
else:
    print("🎉 Everything is perfect — no missing images!")
