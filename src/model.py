# src/model.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small

from src.config import IMG_SIZE, IMG_CHANNELS, NUM_CLASSES, FINAL_ACTIVATION


def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", name=name + "_conv1")(x)
    x = layers.BatchNormalization(name=name + "_bn1")(x)
    x = layers.Activation("relu", name=name + "_act1")(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", name=name + "_conv2")(x)
    x = layers.BatchNormalization(name=name + "_bn2")(x)
    x = layers.Activation("relu", name=name + "_act2")(x)
    return x


def resize_like(x, ref):
    # Resize x to the spatial size of ref using Keras layer
    return layers.Resizing(ref.shape[1], ref.shape[2], interpolation="bilinear")(x)


def build_unetpp():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, IMG_CHANNELS))

    # -----------------------------
    # Encoder: MobileNetV3Small
    # -----------------------------
    base_model = MobileNetV3Small(
        input_tensor=inputs,
        include_top=False,
        weights="imagenet"
    )

    # Stable feature maps from your MobileNetV3 build
    e1 = base_model.get_layer("re_lu").output           # ~128x128
    e2 = base_model.get_layer("re_lu_3").output         # ~64x64
    e3 = base_model.get_layer("re_lu_6").output         # ~32x32
    e4 = base_model.get_layer("re_lu_12").output        # ~16x16
    e5 = base_model.get_layer("activation_17").output   # ~8x8

    # -----------------------------
    # Decoder (Shape-safe, U-Net++ style)
    # -----------------------------
    x5 = conv_block(e5, 256, "x5")

    x4 = conv_block(
        layers.Concatenate()([
            e4,
            resize_like(x5, e4)
        ]),
        256,
        "x4"
    )

    x3 = conv_block(
        layers.Concatenate()([
            e3,
            resize_like(x4, e3)
        ]),
        128,
        "x3"
    )

    x2 = conv_block(
        layers.Concatenate()([
            e2,
            resize_like(x3, e2)
        ]),
        64,
        "x2"
    )

    x1 = conv_block(
        layers.Concatenate()([
            e1,
            resize_like(x2, e1)
        ]),
        32,
        "x1"
    )

    # Final upsampling to full resolution
    x0 = layers.Resizing(IMG_SIZE, IMG_SIZE, interpolation="bilinear")(x1)

    outputs = layers.Conv2D(NUM_CLASSES, 1, padding="same", activation=FINAL_ACTIVATION)(x0)

    model = models.Model(inputs, outputs, name="UNetPlusPlus_MobileNetV3_Safe")

    return model


if __name__ == "__main__":
    model = build_unetpp()
    model.summary()