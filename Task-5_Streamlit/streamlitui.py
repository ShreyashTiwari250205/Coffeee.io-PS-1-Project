# Import Necessary Libraries
import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
import tempfile
import os

# Test Functions 
st.set_page_config(page_title="Image Test", layout="wide")
st.title("Test: Upload and Display")

def convert_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

uploaded_file = st.file_uploader("Choose an image [Allowed : JPG, JPEG, PNG]", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(convert_bgr_to_rgb(image), channels="BGR", caption="Uploaded Image")

# Converting to Grayscale
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Comparing Images (SSIM, MSE)
def compare_images(img1, img2):
    gray1 = convert_to_gray(img1)
    gray2 = convert_to_gray(img2)
    ssim_score, diff = compare_ssim(gray1, gray2, full=True)
    mse_score = np.mean((gray1.astype("float") - gray2.astype("float")) ** 2)
    diff_img = (diff * 255).astype("uint8")
    diff_rgb = cv2.cvtColor(diff_img, cv2.COLOR_GRAY2RGB)
    return round(ssim_score, 4), round(mse_score, 2), diff_rgb


#Object Detection
def detect_objects(img):
    gray = convert_to_gray(img)
    detected_img = img.copy()

    # Load Haar Cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
    right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

    #Face Detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(detected_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(detected_img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = detected_img[y:y+h, x:x+w]

        # Smile detection
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            cv2.putText(roi_color, "Smile", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Left eye detection
        left_eyes = left_eye_cascade.detectMultiScale(roi_gray)
        for (lx, ly, lw, lh) in left_eyes:
            cv2.rectangle(roi_color, (lx, ly), (lx+lw, ly+lh), (0, 255, 0), 2)
            cv2.putText(roi_color, "Left Eye", (lx, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Right eye detection
        right_eyes = right_eye_cascade.detectMultiScale(roi_gray)
        for (rx, ry, rw, rh) in right_eyes:
            cv2.rectangle(roi_color, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)
            cv2.putText(roi_color, "Right Eye", (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return detected_img

# Convert NumPy array to JPG byte buffer for download
def to_jpg_bytes(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return buffer.tobytes()


# Streamlit UI
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
    ssim_score, mse_score, diff_rgb = compare_images(img1, img2)
    st.write(f"**SSIM:** {ssim_score}")
    st.write(f"**MSE:** {mse_score}")

    st.image([convert_bgr_to_rgb(img1), convert_bgr_to_rgb(img2), diff_rgb], caption=["Image 1", "Image 2", "Difference Image"], width=300)

    st.subheader("FACE DETECTION USING HAAR CASCADES")
    detected_img = detect_objects(img1)
    st.image(convert_bgr_to_rgb(detected_img), caption="Face Detection in Image 1", width=400)

    # Download Results
    st.subheader("Download Results")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("Download Diff Image", data=to_jpg_bytes(diff_rgb), file_name="difference.jpg")
    with col_dl2:
        st.download_button("Download Detection Image", data=to_jpg_bytes(detected_img), file_name="detection.jpg")
else:
    st.info("Please upload both images to proceed.")
