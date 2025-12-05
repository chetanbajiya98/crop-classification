import streamlit as st
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(page_title="Crop Classification App", layout="wide")

# ----------------------------------------------------------
# LOAD MODEL + FEATURE LIST
# ----------------------------------------------------------
@st.cache_resource
def load_model_files():
    model = joblib.load("wheat_rf_model.joblib")
    feature_list = joblib.load("wheat_rf_features.joblib")
    return model, feature_list

model, feature_list = load_model_files()

st.title("🌾 Crop Classification (Wheat vs Non-Wheat)")
st.write("Upload your Sentinel-2 preprocessed **Top-10 feature stack TIFF** for classification.")

# ----------------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------------
uploaded_file = st.file_uploader("Upload GeoTIFF", type=["tif", "tiff"])

if uploaded_file is not None:
    with rasterio.open(uploaded_file) as src:
        profile = src.profile
        image = src.read()  # Shape: (bands, H, W)

    st.success(f"Loaded raster with {image.shape[0]} bands")

    # ------------------------------------------------------
    # MATCH BANDS TO FEATURES
    # ------------------------------------------------------
    band_names = feature_list  # The exact features used in training

    if len(band_names) != image.shape[0]:
        st.error("Band mismatch! Uploaded TIFF does not match expected features.")
        st.stop()

    # Reshape for model input
    H, W = image.shape[1], image.shape[2]
    reshaped = image.reshape(image.shape[0], -1).T  # (pixels, bands)

    st.info(f"Running classification on {reshaped.shape[0]:,} pixels...")

    # ------------------------------------------------------
    # MODEL PREDICTION
    # ------------------------------------------------------
    preds = model.predict(reshaped)
    classified = preds.reshape(H, W)

    # ------------------------------------------------------
    # DISPLAY OUTPUT
    # ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("viridis")
    ax.imshow(classified, cmap=cmap)
    ax.set_title("Classification Map")
    ax.axis("off")
    st.pyplot(fig)

    # ------------------------------------------------------
    # SAVE OUTPUT TIFF
    # ------------------------------------------------------
    out_file = "classified_output.tif"
    profile.update(dtype=rasterio.uint8, count=1)

    with rasterio.open(out_file, "w", **profile) as dst:
        dst.write(classified.astype("uint8"), 1)

    with open(out_file, "rb") as f:
        st.download_button("⬇️ Download Classified TIFF", f, file_name="classified_map.tif")
