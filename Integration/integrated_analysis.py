import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt

# === Set Up Output Directory ===
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

def integrated_analysis(img_path1, img_path2, haar_cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):
    # --- Load and preprocess images ---
    img1 = cv2.imread("Sample_Images/image_1.jpg")
    img2 = cv2.imread("Sample_Images/image_2.jpg")
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # === Step 1: SSIM and MSE ===
    ssim_score, diff = compare_ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    mse_score = np.mean((gray1.astype("float") - gray2.astype("float")) ** 2)

    # Save difference image
    diff_path = os.path.join(output_dir, "difference_image.jpg")
    cv2.imwrite(diff_path, diff)

    # === Step 2: Object Detection using Haar Cascade ===
    detection_img = img1.copy()
    haar = cv2.CascadeClassifier(haar_cascade_path)
    faces = haar.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save detection image
    detection_path = os.path.join(output_dir, "detection_output.jpg")
    cv2.imwrite(detection_path, detection_img)

    # === Step 3: Display All Outputs ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(diff, cmap='gray')
    axes[1].set_title(f"Difference\nSSIM: {ssim_score:.4f} | MSE: {mse_score:.2f}")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Object Detection Output")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return {
        "SSIM Score": round(ssim_score, 4),
        "MSE Score": round(mse_score, 2),
        "Difference Image Path": diff_path,
        "Detection Image Path": detection_path
    }

# === Example usage ===
if __name__ == "__main__":
    img1 = "sample_images/face_clear.jpg"
    img2 = "sample_images/face_occluded.jpg"

    result = integrated_analysis(img1, img2)

    print("\n--- Results ---")
    for k, v in result.items():
        print(f"{k}: {v}")
