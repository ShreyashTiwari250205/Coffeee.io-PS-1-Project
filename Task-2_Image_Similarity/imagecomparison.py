# Import necessary Libraries
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Function to calculate MSE
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# Function to calculate SSIM
def compare_images():

    # Read Images
    img1=cv2.imread('Sample_Images/face_with_glasses.jpg')
    img2=cv2.imread('Sample_Images/face_without_glasses.jpg')

    # Resizing Images (Necessary for comparison)
    img1 = cv2.resize(img1,(500,500))
    img2 = cv2.resize(img2,(500,500))

    # Converting Images to Grayscale (Faster)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculating SSIM
    ssim_score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype('uint8')

    # Calculating MSE
    error = mse(gray1, gray2)

    print(f"SSIM: {ssim_score:.4f}")
    print(f"Mean Squared Error: {error}")

    # Thresholding and Contouring
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours,_ =  cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2. CHAIN_APPROX_SIMPLE)
    
    img_diff = img2.copy()
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img_diff, (x,y), (x+w, y+h), (0,0,255),2)
    
    # Showing Original and Final Image
    cv2.imshow('1', img1)
    cv2.imshow('2', img2)
    cv2.imshow('differences', img_diff)
    img_diff = (img_diff * 255).astype("uint8")
    # Saving Images
    cv2.imwrite("Image_Similarity/diff_image.jpg", img_diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
compare_images()

