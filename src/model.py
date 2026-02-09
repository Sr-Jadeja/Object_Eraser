# src/model.py

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import IMG_SIZE, IMG_CHANNELS, NUM_CLASSES, FINAL_ACTIVATION


def conv_block(x, filters, kernel_size=3):
    """Conv2D -> BatchNorm -> ReLU"""
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def upsample_block(x, filters):
    """Upsample -> two conv blocks"""
    x = layers.UpSampling2D((2, 2))(x)
    x = conv_block(x, filters)
    x = conv_block(x, filters)
    return x


def build_model():
    """
    MobileNetV3Small encoder + simple U-Net style decoder (no fragile skips)
    """

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS))

    # Encoder
    base_model = tf.keras.applications.MobileNetV3Small(
        input_tensor=inputs,
        include_top=False,
        weights="imagenet"
    )

    encoder_output = base_model.output  # ~8x8 feature map

    # Decoder
    x = conv_block(encoder_output, 512)

    x = upsample_block(x, 256)  # 8 -> 16
    x = upsample_block(x, 128)  # 16 -> 32
    x = upsample_block(x, 64)   # 32 -> 64
    x = upsample_block(x, 32)   # 64 -> 128
    x = upsample_block(x, 16)   # 128 -> 256

    # Output layer: 1-channel mask
    outputs = layers.Conv2D(
        NUM_CLASSES,
        kernel_size=1,
        padding="same",
        activation=FINAL_ACTIVATION
    )(x)

    model = models.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()