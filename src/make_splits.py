import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/train.csv")
print("Total available with images:", len(df))

# 70% train, 15% val, 15% test (stratified by label)
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
)

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
