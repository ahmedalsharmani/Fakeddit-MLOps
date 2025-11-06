import tensorflow as tf
from tensorflow.keras import layers as L, models as M

def build_text_branch(max_tokens=40000, seq_len=160, embed_dim=200):
    text_in = L.Input(shape=(), dtype=tf.string, name="text")
    text_vec = L.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=seq_len,
        name="TextVec",
    )
    x = text_vec(text_in)
    x = L.Embedding(max_tokens, embed_dim, mask_zero=True)(x)
    x = L.Conv1D(256, 5, activation="relu")(x)
    x = L.GlobalMaxPooling1D()(x)
    x = L.Dropout(0.2)(x)
    tfeat = L.Dense(256, activation="relu")(x)
    tfeat = L.BatchNormalization(name="bn_text")(tfeat)  # normalize feature scale
    return text_in, text_vec, tfeat

def build_image_branch():
    img_in = L.Input(shape=(224, 224, 3), name="image")
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=img_in
    )
    base.trainable = False  # we'll unfreeze a few layers later during fine-tuning
    y = L.GlobalAveragePooling2D()(base.output)
    y = L.Dropout(0.2)(y)
    ifeat = L.Dense(256, activation="relu")(y)
    ifeat = L.BatchNormalization(name="bn_image")(ifeat)  # normalize feature scale
    return img_in, base, ifeat

def build_model(num_classes=2):
    text_in, text_vec, tfeat = build_text_branch()
    img_in, base, ifeat = build_image_branch()
    z = L.Concatenate(name="fusion_concat")([tfeat, ifeat])
    z = L.Dense(256, activation="relu")(z)
    z = L.Dropout(0.5)(z)
    out = L.Dense(num_classes, activation="softmax", name="pred")(z)
    model = M.Model(inputs={"text": text_in, "image": img_in}, outputs=out)
    # expose the base for fine-tuning via name
    base._name = "resnet50_base"
    return model, text_vec
