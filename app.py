import streamlit as st
import os
import gdown
import torch
from PIL import Image

# -------------------------------
# 📌 CONFIG
# -------------------------------
MODEL_PATH = "best_model.pth"
FILE_ID = "1LCfciCjUA77qF_gwL5OyOxES2AAo0HCM"  # your drive file id

# -------------------------------
# 📌 DOWNLOAD MODEL
# -------------------------------
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model... please wait ⏳")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully ✅")
    except Exception as e:
        st.error(f"Download failed: {e}")

# -------------------------------
# 📌 LOAD MODEL (SAFE)
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = torch.load(MODEL_PATH, map_location="cpu")
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# -------------------------------
# 📌 UI
# -------------------------------
st.title("Cross-Lingual Visual Question Answering System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
question = st.text_input("Ask a question (English or Telugu)")

# -------------------------------
# 📌 PREDICTION
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if question:
        try:
            if model is not None:
                # 🔥 TODO: replace with your actual model logic

                # Example placeholder:
                answer = "Model Answer (placeholder)"

            else:
                # fallback demo mode
                answer = "Demo Answer"

        except Exception as e:
            st.error(f"Prediction error: {e}")
            answer = "Error during prediction"

        st.subheader("Answer:")
        st.write(answer)
