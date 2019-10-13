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
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(frame, [box], 0, (0, 0, 255))

        # finally, get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        # convert all values to int
        center = (int(x), int(y))
        radius = int(radius)
        # and draw the circle in blue
        frame = cv2.circle(frame, center, radius, (255, 0, 0), 2)

    print(len(contours))
    cv2.drawContours(frame, contours, -1, (255, 255, 0), 1)

    cv2.imshow("contours", frame)

    cv2.imshow("contours", frame)

    cv2.imshow('hsv', hsv)
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()