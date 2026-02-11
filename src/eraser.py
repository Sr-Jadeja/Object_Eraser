# src/eraser.py

import os
import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return mask


def refine_mask(mask):
    """
    Improve the predicted mask:
    - Strong threshold to remove weak false positives
    - Keep only the largest connected component (assume it's the person)
    - Morphology to clean holes and noise
    - Slight feathering for smoother edges
    """

    # 1) Strong threshold (more conservative)
    _, mask_bin = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)

    # 2) Keep only largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

    if num_labels > 1:
        # Skip background label 0
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_largest = np.zeros_like(mask_bin)
        mask_largest[labels == largest_label] = 255
    else:
        mask_largest = mask_bin.copy()

    # 3) Morphology to clean edges
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_largest, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4) Feather edges slightly
    mask_blur = cv2.GaussianBlur(mask_clean, (9, 9), 0)

    # Re-binarize after blur
    _, mask_final = cv2.threshold(mask_blur, 128, 255, cv2.THRESH_BINARY)

    return mask_final


def main():
    # -----------------------------
    # Paths
    # -----------------------------
    input_image_path = "test.jpg"
    mask_path = "pred_mask.png"
    output_path = "erased.png"

    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    print("Loading image:", input_image_path)
    image = load_image(input_image_path)

    print("Loading mask:", mask_path)
    mask = load_mask(mask_path)

    print("Refining mask...")
    refined_mask = refine_mask(mask)

    # Resize mask to match original image size
    h, w = image.shape[:2]
    refined_mask = cv2.resize(refined_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    print("Running inpainting (object removal)...")
    # Use Navier-Stokes method for more structure-preserving fill
    erased = cv2.inpaint(
        image,
        refined_mask,
        inpaintRadius=2,
        flags=cv2.INPAINT_NS
    )

    print("Saving result to:", output_path)
    cv2.imwrite(output_path, erased)

    print("Done! Object erased image saved as:", output_path)


if __name__ == "__main__":
    main()