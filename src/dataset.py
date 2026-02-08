# src/dataset.py

import os
import cv2
import numpy as np
import tensorflow as tf

from src.config import IMG_SIZE, BATCH_SIZE


def load_image(image_path):
    """
    Load and preprocess an image:
    - Read from disk
    - Resize to (IMG_SIZE, IMG_SIZE)
    - Normalize to range [0, 1]
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    return image


def load_mask(mask_path):
    """
    Load and preprocess a mask:
    - Read from disk (grayscale)
    - Resize to (IMG_SIZE, IMG_SIZE)
    - Convert to binary mask (0 or 1)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    # Convert mask to 0 or 1
    mask = mask.astype(np.float32)
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)  # (H, W) -> (H, W, 1)

    return mask


def load_image_mask_pair(image_path, mask_path):
    """
    Load one image and its corresponding mask.
    This function will be wrapped by TensorFlow later.
    """
    image = load_image(image_path.decode("utf-8"))
    mask = load_mask(mask_path.decode("utf-8"))
    return image, mask


def tf_load_image_mask(image_path, mask_path):
    """
    TensorFlow wrapper around the Python loading function.
    """
    image, mask = tf.numpy_function(
        load_image_mask_pair,
        [image_path, mask_path],
        [tf.float32, tf.float32]
    )

    image.set_shape((IMG_SIZE, IMG_SIZE, 3))
    mask.set_shape((IMG_SIZE, IMG_SIZE, 1))

    return image, mask


def get_dataset(images_dir, masks_dir, shuffle=True):
    """
    Create a tf.data.Dataset from image and mask folders.
    """
    image_files = sorted([
        os.path.join(images_dir, fname) for fname in os.listdir(images_dir)
    ])
    mask_files = sorted([
        os.path.join(masks_dir, fname) for fname in os.listdir(masks_dir)
    ])

    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    dataset = dataset.map(tf_load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset