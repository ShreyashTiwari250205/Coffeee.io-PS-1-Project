import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
import tempfile
import os

# === Helper functions ===
st.set_page_config(page_title="Image Test", layout="wide")
st.title("Test: Upload and Display")

uploaded_file = st.file_uploader("Choose an image [Allowed : JPG, JPEG, PNG]", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Uploaded Image")


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def compare_images(img1, img2):
    gray1 = convert_to_gray(img1)
    gray2 = convert_to_gray(img2)
    ssim_score, diff = compare_ssim(gray1, gray2, full=True)
    mse_score = np.mean((gray1.astype("float") - gray2.astype("float")) ** 2)
    diff_img = (diff * 255).astype("uint8")
    return round(ssim_score, 4), round(mse_score, 2), diff_img

def detect_faces(img):
    gray = convert_to_gray(img)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    detected_img = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(detected_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return detected_img

def to_jpg_bytes(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return buffer.tobytes()

# === Streamlit UI ===

st.title("IMAGE SIMILARITY AND OBJECT DETECTION TOOL UI")
st.write("Upload two images to compare and detect faces using Haar Cascade.")

col1, col2 = st.columns(2)

with col1:
    img1_file = st.file_uploader("Upload Image 1 [Allowed : JPG, JPEG, PNG]", type=["jpg", "jpeg", "png"], key="img1")

with col2:
    img2_file = st.file_uploader("Upload Image 2 [Allowed : JPG, JPEG, PNG]", type=["jpg", "jpeg", "png"], key="img2")

if img1_file and img2_file:
    img1 = cv2.imdecode(np.frombuffer(img1_file.read(), np.uint8), 1)
    img2 = cv2.imdecode(np.frombuffer(img2_file.read(), np.uint8), 1)

    st.subheader("SIMILARITY ANALYSIS")
    ssim_score, mse_score, diff_img = compare_images(img1, img2)
    st.write(f"**SSIM:** {ssim_score}")
    st.write(f"**MSE:** {mse_score}")

    st.image([img1, img2, diff_img], caption=["Image 1", "Image 2", "Difference Image"], width=300)

    st.subheader("FACE DETECTION USING HAAR CASCADES")
    detected_img = detect_faces(img1)
    st.image(detected_img, caption="Face Detection in Image 1", width=400)

    # === Download options ===
    st.subheader("Download Results")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("Download Diff Image", data=to_jpg_bytes(diff_img), file_name="difference.jpg")
    with col_dl2:
        st.download_button("Download Detection Image", data=to_jpg_bytes(detected_img), file_name="detection.jpg")
else:
    st.info("Please upload both images to proceed.")
