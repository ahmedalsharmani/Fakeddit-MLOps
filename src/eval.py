import tensorflow as tf, pandas as pd, numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from src.dataset_tfdata import make_ds
from src.model_textcnn_resnet import build_model

train_csv = "data/train.csv"
test_csv  = "data/test.csv"

# Rebuild architecture and re-adapt TextVectorization
model, text_vec = build_model(num_classes=2)
texts = pd.read_csv(train_csv)["text"].astype(str).tolist()
text_vec.adapt(tf.data.Dataset.from_tensor_slices(texts).batch(256))

# Load trained weights
model.load_weights("fusion_model.weights.h5")

# Evaluate
test_ds = make_ds(test_csv, batch=32)
y_true, y_pred = [], []
for x, y in test_ds:
    p = model.predict(x, verbose=0)
    y_true.extend(y.numpy().tolist())
    y_pred.extend(np.argmax(p, axis=1).tolist())

print("Macro-F1:", f1_score(y_true, y_pred, average="macro"))
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=3))
