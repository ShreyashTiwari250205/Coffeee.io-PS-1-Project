# 🖼️ Image Similarity & Object Detection Tool

A **Streamlit-based web application** that allows users to:

- Compare two uploaded images using **SSIM (Structural Similarity Index)** and **MSE (Mean Squared Error)**.
- Detect **faces, smiles, and eyes (left & right)** using **Haar Cascade classifiers**.
- Visualize differences and object annotations on images.
- Download annotated and difference images.

------------------------------------------------------------------------------------------------------------------------------------------------------

## 📌 Features
Features and Description:
a) Image Upload : Upload two images in JPG, JPEG or PNG in Streamlit made UI.
b) Image Comparison : Calculate SSIM and MSE and display visual difference in reference and captured image.
c) Object Detection : Detects and Labels - Faces, Smiles and Eyes in image.
d) Downloads : Can download annotated detection and comparison results through Streamlit UI.
e) Built Using : Streamlit, OpenCV, NumPy, Scikit-Learn, Matplotlib, Pandas, Python and VS Code.

-------------------------------------------------------------------------------------------------------------------------------------------------------

### File Structure

-> Task-1_Setup_and_Research
    -> Sample_Images
        -> contrast-cucumbers.png
        -> face_with_glasses.jpg
        -> face_without_glasses.jpg
        -> image1.png
        -> image2.png
        -> original-cucumbers.png
        -> ref_image1.webp
        -> ref_image2.webp
    -> environmentcheck.py
    -> setup_screenshot.png
    -> setup_with_images_proof.png
-> Task-2_Image_Similarity
    -> Sample_Images
        -> contrast-cucumbers.png
        -> face_with_glasses.jpg
        -> face_without_glasses.jpg
        -> image1.png
        -> image2.png
        -> original-cucumbers.png
    -> imagecomparison.py
    -> output-1.png
    -> output-2.png
    -> similarity_result.png
-> Task-3_Object_Detection
    -> Output.png
    -> Output_2.png
    -> haar_frontalface_default.xml
    -> haarcascade_lefteye_2splits.xml
    -> haarcascade_righteye_2splits.xml
    -> haarcascade_smile.xml
    -> image.jpeg
    -> image_2.jpeg
    -> image_3.jpeg
    ->objectdetection.py
    ->output_3.png
-> Task-4_Integration
    -> Sample_Images
        -> Image_1.jpg
        -> Image_2.jpg
        -> Image_3.jpg
        -> Image_4.jpg
    -> output_images
        -> detection_output.jpg
        -> detection_output_1.jpg
        -> detection_output_2.jpg
        -> difference_image.jpg
        -> difference_image_1.jpg
        -> difference_image_2.jpg
    -> haar_frontalface_default.xml
    -> haarcascade_lefteye_2splits.xml
    -> haarcascade_righteye_2splits.xml
    -> haarcascade_smile.xml
    -> integrated analysis.py
-> Task-5_Streamlit
    -> streamlitui.py
-> Task-6_Outputs
    -> UI_Output_2.png
    -> UI_Output_2.png
    
