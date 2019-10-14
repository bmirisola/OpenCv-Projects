import cv2
import numpy as np
import Constants

face_cascade = cv2.CascadeClassifier('/home/pi/python_projects/OpenCV/data/haarcascades/haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(Constants.CAPTURE_SOURCE_ID)
cap.set (3,640)
cap.set(4,480)

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
