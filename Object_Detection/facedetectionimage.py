import cv2

face_cascade = cv2.CascadeClassifier('haar_frontalface_default.xml')

img = cv2.imread('face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4) #vector of rectangles

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)

cv2.imshow('detected', img)
cv2.waitKey(0)
 