import cv2
import numpy
import torch
import streamlit
from skimage.metrics import structural_similarity

print("All libraries installed and imported successfully!")

img1 = cv2.imread('Sample_Images/face_with_glasses.jpg')
img2 = cv2.imread('Sample_Images/face_without_glasses.jpg')
img3 = cv2.imread('Sample_Images/ref_image1.webp')
img4 = cv2.imread('Sample_Images/ref_image2.webp')

if img1 is None or img2 is None or img3 is None or img4 is None:
    print("Error: One or more images could not be read. Check file paths or extensions.")
else:
    print("All images read successfully!")