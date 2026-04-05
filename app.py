import streamlit as st
import os
import gdown
from PIL import Image

# -------------------------------
# 📌 DOWNLOAD MODEL FROM DRIVE
# -------------------------------
MODEL_PATH = "best_model.pth"
FILE_ID = "1LCfciCjUA77qF_gwL5OyOxES2AAo0HCM"  # your file id

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model... please wait ⏳")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully ✅")
    except Exception as e:
        st.error(f"Download failed: {e}")

# -------------------------------
# 📌 IMPORT MODEL FUNCTION
# -------------------------------
try:
    from model_utils import predict
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Model import failed: {e}")
    MODEL_LOADED = False

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
            # Save uploaded image temporarily
            temp_path = "temp.jpg"
            image.save(temp_path)

            if MODEL_LOADED:
                # 🔥 REAL MODEL CALL
                answer = predict(temp_path, question)
            else:
                # fallback if model fails
                answer = "Demo Answer"

        except Exception as e:
            st.error(f"Prediction error: {e}")
            answer = "Error during prediction"

        st.subheader("Answer:")
        st.write(answer)
