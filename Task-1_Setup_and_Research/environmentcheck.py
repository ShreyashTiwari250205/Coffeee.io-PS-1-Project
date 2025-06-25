# Importing necessary libraries
import cv2
import numpy
import torch
import streamlit
from skimage.metrics import structural_similarity

# Check if libraries were imported successfully
print("All libraries installed and imported successfully!")

# Reading Sample Images
img1 = cv2.imread('Sample_Images/face_with_glasses.jpg')
img2 = cv2.imread('Sample_Images/face_without_glasses.jpg')
img3 = cv2.imread('Sample_Images/ref_image1.webp')
img4 = cv2.imread('Sample_Images/ref_image2.webp')
img5 = cv2.imread('Sample_Images/image1.png')
img6 = cv2.imread('Sample_Images/image2.png')
img7 = cv2.imread('Sample_Images/original-cucumbers.png')
img8 = cv2.imread('Sample_Images/contrast-cucumbers.png')

# Checking if all images were read successfully
if img1 is None or img2 is None or img3 is None or img4 is None or img5 is None or img6 is None or img7 is None or img8 is None:
    print("Error: One or more images could not be read. Check file paths or extensions.")
else:
    print("All images read successfully!")