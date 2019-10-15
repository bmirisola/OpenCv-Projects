import cv2
import Constants

# cap = cv2.VideoCapture(Constants.CAPTURE_SOURCE_ID)
# cap.set (3,Constants.CAMERA_WIDTH)
# cap.set(4,Constants.CAMERA_HEIGHT)
#
# while (True):
#     ret, frame = cap.read()
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(Constants.PROTOTEXT, Constants.MODEL)

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(Constants.IMAGE)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))