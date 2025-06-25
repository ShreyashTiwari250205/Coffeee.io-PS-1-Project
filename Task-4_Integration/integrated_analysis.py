# Import Necessary Libraries
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt

#Set Up Output Directory
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Function to Calculate Similarity and Detect Objects
def integrated_analysis(img_path1, img_path2):
    # Read the Images
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # Convert to Grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # First, calculate SSIM and MSE
    ssim_score, diff = compare_ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    mse_score = np.mean((gray1.astype("float") - gray2.astype("float")) ** 2)

    # Save difference image
    diff_path = os.path.join(output_dir, "difference_image.jpg")
    cv2.imwrite(diff_path, diff)

    # Detect Faces, Eyes and Smile using Haar Cascades
    detection_img = img1.copy()

    # Load all Haar cascades
    face_cascade = cv2.CascadeClassifier('haar_frontalface_default.xml')
    left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

    # Detect Faces
    faces = face_cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Face rectangle and label
        cv2.rectangle(detection_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(detection_img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        roi_gray = gray1[y:y + h, x:x + w]
        roi_color = detection_img[y:y + h, x:x + w]

        # Left Eye Detection
        left_eyes = left_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        for (ex, ey, ew, eh) in left_eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(roi_color, "Eye", (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Right Eye Detection
        right_eyes = right_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        for (ex, ey, ew, eh) in right_eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 200, 200), 2)
            cv2.putText(roi_color, "Eye", (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)

        # Smile Detection
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            cv2.putText(roi_color, "Smile", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Save detection image
    detection_path = os.path.join(output_dir, "detection_output.jpg")
    cv2.imwrite(detection_path, detection_img)

    # Display All Outputs using MATPLOTLIB
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(diff, cmap='gray')
    axes[1].set_title(f"Difference\nSSIM: {ssim_score:.4f} | MSE: {mse_score:.2f}")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Detection Output")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return {
        "SSIM Score": round(ssim_score, 4),
        "MSE Score": round(mse_score, 2),
        "Difference Image Path": diff_path,
        "Detection Image Path": detection_path
    }

# Using the code
if __name__ == "__main__":
    img1 = "Sample_Images/Image_3.jpg"
    img2 = "Sample_Images/Image_4.jpg"

    result = integrated_analysis(img1, img2)

    print("\n--- Results ---")
    for k, v in result.items():
        print(f"{k}: {v}")
