import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "weights/model.keras"
IMG_SIZE = 256  # must match training size

# ----------------------------
# Load model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# ----------------------------
# Preprocess image
# ----------------------------
def preprocess_image_pil(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)
    return img_input, img_resized

# ----------------------------
# Postprocess mask
# ----------------------------
def postprocess_mask(pred):
    mask = pred[0, :, :, 0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

# ----------------------------
# UI
# ----------------------------
st.title("Human Segmentation System")
st.write("Upload an image and the model will generate a segmentation mask.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)

    # Preprocess
    img_input, img_resized = preprocess_image_pil(pil_img)

    # Predict
    with st.spinner("Running model..."):
        pred = model.predict(img_input)

    mask = postprocess_mask(pred)

    # Create overlay
    overlay = img_resized.copy()
    overlay[mask == 255] = [255, 0, 0]  # red mask
    blended = cv2.addWeighted(img_resized, 0.7, overlay, 0.3, 0)

    # ----------------------------
    # Layout: 3 columns (side by side)
    # ----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.image(pil_img, use_container_width=True)

    with col2:
        st.subheader("Predicted Mask")
        st.image(mask, use_container_width=True)

    with col3:
        st.subheader("Overlay")
        st.image(blended, use_container_width=True)