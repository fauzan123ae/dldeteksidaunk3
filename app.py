import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# -----------------------------------------
# KONFIGURASI DASAR
# -----------------------------------------

CLASS_NAMES = ['blight', 'healthy', 'rust']
IMG_SIZE = (128, 128)
MODEL_PATH = "corn_leaf_cnn_model.h5"

# -----------------------------------------
# FUNGSI: Load Model
# -----------------------------------------

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model tidak ditemukan di {MODEL_PATH}")
        return None
    return load_model(MODEL_PATH)

# -----------------------------------------
# FUNGSI: Prediksi Gambar
# -----------------------------------------

def predict_image(img, model):
    img = img.resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    pred_index = np.argmax(prediction)
    pred_label = CLASS_NAMES[pred_index]
    confidence = float(prediction[pred_index])

    return pred_label, confidence

# -----------------------------------------
# STREAMLIT UI
# -----------------------------------------

st.title("🌽 Deteksi Penyakit Daun Jagung")
st.write("Upload gambar daun jagung, lalu sistem akan mendeteksi penyakit berdasarkan model yang dilatih dari dataset.")

model = load_cnn_model()

uploaded_file = st.file_uploader("📤 Upload Gambar Daun Jagung", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Gambar yang Diupload", use_column_width=True)

    if st.button("🔍 Prediksi Gambar"):
        with st.spinner("Sedang memproses..."):
            label, confidence = predict_image(image, model)
            st.success(f"✅ Hasil Prediksi: **{label}** ({confidence*100:.2f}%)")
