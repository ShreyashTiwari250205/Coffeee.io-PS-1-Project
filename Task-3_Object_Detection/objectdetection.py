# Import necessary libraries
import cv2

# Load image
img = cv2.imread('image_3.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier('haar_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.2, 3)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    # Detect eyes in face region
    left_eyes = left_eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in left_eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.putText(roi_color, "Eye", (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    right_eyes = right_eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in right_eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.putText(roi_color, "Eye", (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Detect smiles in face region
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=20)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
        cv2.putText(roi_color, "Smile", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Display output image
cv2.imshow('Detection Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
