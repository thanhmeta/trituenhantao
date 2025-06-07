import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import os
import gdown

# ================== CẤU HÌNH ==================
MODEL_PATH = "plant_disease_model_update.h5"
CLASS_INDEX_PATH = "class_indices.json"
FILE_ID = "1UEheXekm6EakPq8COAKumPoHXQuOsetP"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ================== TẢI MÔ HÌNH ==================
if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Đang tải mô hình từ Google Drive..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

# Load mô hình
model = tf.keras.models.load_model(MODEL_PATH)

# Load mapping class index -> class name
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

# ================== GIAO DIỆN ==================
st.set_page_config(page_title="Phân loại bệnh lá cà chua", layout="centered")

# === Sidebar ===
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("### 📂 Tải ảnh lá cà chua")
    uploaded_file = st.file_uploader("Chọn ảnh (jpg/png)...", type=["jpg", "jpeg", "png"])

# === Main Area ===
st.markdown("<h1 style='text-align: center;'>🍅 Phân loại bệnh lá cà chua bằng VGG16</h1>", unsafe_allow_html=True)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Ảnh đã chọn", use_container_width=True)

        # Tiền xử lý ảnh
        img_resized = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    with col2:
        with st.spinner("🔍 Đang dự đoán..."):
            prediction = model.predict(img_array)
            predicted_index = int(np.argmax(prediction))
            predicted_class = index_to_class[predicted_index]
            confidence = float(np.max(prediction)) * 100

        st.success("✅ Dự đoán hoàn tất!")
        st.markdown(f"**🩺 Bệnh:** `{predicted_class}`")
        st.markdown(f"**📊 Độ tin cậy:** `{confidence:.2f}%`")
else:
    st.info("📌 Vui lòng chọn ảnh để phân loại.")
