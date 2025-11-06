"""
Grad-CAM explainability for image branch.
"""
import tensorflow as tf
import numpy as np
from pathlib import Path
from .model_textcnn_resnet import build_model

def get_gradcam(model, img_array, layer_name='conv5_block3_out'):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array])
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs[0], weights)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = tf.image.resize(cam[..., np.newaxis], (224, 224)).numpy()
    return cam
