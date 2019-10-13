import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
#os.system("sudo scripts/configure.sh")
cap.set (3,640)
cap.set(4,480)

def print_hsv_at_coord(event, x, y, empty, data):
    global hsv
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print (hsv[y, x])

cv2.namedWindow('hsv')
cv2.setMouseCallback('hsv', print_hsv_at_coord)

lower_range = np.array([103, 170, 175], dtype=np.uint8)
upper_range = np.array([108, 200, 200], dtype=np.uint8)

while (True):
    ret, frame = cap.read()

    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    normalized_frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    #blurred = cv2.GaussianBlur(normalized_frame, (11, 11), 0)

    hsv = cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)

    cv2.imshow('hsv', hsv)
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()