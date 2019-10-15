import cv2
import numpy as np
import Constants

cap = cv2.VideoCapture(Constants.CAPTURE_SOURCE_ID)
cap.set (3,Constants.CAMERA_WIDTH)
cap.set(4,Constants.CAMERA_HEIGHT)

while (True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()