# src/train_fusion_align.py
import tensorflow as tf, pandas as pd
from src.dataset_tfdata import make_ds
from src.model_textcnn_resnet import build_model

# =======================================
#   1️⃣  Utility Functions
# =======================================

def l2n(x):
    return tf.math.l2_normalize(x, axis=-1)

def info_nce_loss(tz, iz, temperature=0.07):
    logits = tf.matmul(tz, iz, transpose_b=True) / temperature
    labels = tf.range(tf.shape(logits)[0])
    loss_t = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    loss_i = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
    return 0.5 * (tf.reduce_mean(loss_t) + tf.reduce_mean(loss_i))

# =======================================
#   2️⃣  Dataset + Config
# =======================================

train_csv = "data/train.csv"
val_csv   = "data/val.csv"

EPOCHS_HEAD = 3
EPOCHS_FT   = 2
BATCH = 32

W_CLF    = 1.0
W_ALIGN  = 0.1
W_AUXIMG = 0.1

# =======================================
#   3️⃣  Build Model
# =======================================

model, text_vec = build_model(num_classes=2)

# Adapt TextVectorization layer
texts = pd.read_csv(train_csv)["text"].astype(str).tolist()
text_vec.adapt(tf.data.Dataset.from_tensor_slices(texts).batch(256))

train_ds = make_ds(train_csv, batch=BATCH, shuffle=True)
val_ds   = make_ds(val_csv, batch=BATCH)

optimizer = tf.keras.optimizers.Adam(1e-3)
clf_loss  = tf.keras.losses.SparseCategoricalCrossentropy()

# ---- Sub-model to extract embeddings ----
embedder = tf.keras.Model(
    inputs=model.inputs,
    outputs={
        "text": model.get_layer("bn_text").output,
        "image": model.get_layer("bn_image").output,
    }
)

# =======================================
#   4️⃣  Training Step
# =======================================
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        loss_main = clf_loss(y, out)

        emb = embedder(x, training=True)
        text_emb, img_emb = l2n(emb["text"]), l2n(emb["image"])
        loss_align = info_nce_loss(text_emb, img_emb)

        batch_size = tf.shape(x["text"])[0]
        x_img_only = {"image": x["image"], "text": tf.zeros((batch_size,), dtype=tf.string)}
        out_img_only = model(x_img_only, training=True)
        loss_img_aux = clf_loss(y, out_img_only)

        total_loss = (W_CLF * loss_main +
                      W_ALIGN * loss_align +
                      W_AUXIMG * loss_img_aux)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss

# =======================================
#   5️⃣  Validation Step (manual)
# =======================================
@tf.function
def val_step(x, y):
    out = model(x, training=False)
    preds = tf.argmax(out, axis=1, output_type=tf.int64)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, tf.cast(y, tf.int64)), tf.float32))
    return acc

# =======================================
#   6️⃣  Training Loop
# =======================================
def run_stage(epochs, unfreeze=False, lr=1e-3):
    global optimizer
    optimizer = tf.keras.optimizers.Adam(lr)

    if unfreeze:
        # try to find the image backbone automatically
        resnet = None
        for layer in model.layers:
            if "conv1_conv" in layer.name:  # first ResNet layer
                resnet = model
                break

        if resnet is not None:
            print("🔓 Unfreezing last 40 convolutional layers in ResNet backbone")
            # Freeze most layers except last 40
            for l in resnet.layers[:-40]:
                l.trainable = False
            for l in resnet.layers[-40:]:
                l.trainable = True
        else:
            print("⚠️  Could not find ResNet backbone; proceeding with all trainable layers.")



    for ep in range(1, epochs + 1):
        print(f"\nEpoch {ep}/{epochs}")
        total_loss = 0.0
        n_steps = 0
        for x, y in train_ds:
            total_loss += train_step(x, y)
            n_steps += 1
        avg_loss = float(total_loss / n_steps)

        # compute validation accuracy manually
        val_accs = []
        for x, y in val_ds:
            val_accs.append(val_step(x, y))
        val_acc = float(tf.reduce_mean(val_accs))

        print(f"  Train loss: {avg_loss:.4f} | Val acc: {val_acc:.3f}")

# =======================================
#   7️⃣  Run Training
# =======================================
print("\n🧩 Stage 1: Train fusion head (ResNet frozen)")
run_stage(EPOCHS_HEAD, unfreeze=False, lr=1e-3)

print("\n🖼️ Stage 2: Fine-tune with alignment")
run_stage(EPOCHS_FT, unfreeze=True, lr=1e-4)

model.save_weights("fusion_model_aligned.weights.h5")
print("\n✅ Saved weights -> fusion_model_aligned.weights.h5")
