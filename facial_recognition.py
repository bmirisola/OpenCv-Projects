import cv2
import Constants

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = Constants.HAARCASCADE_LOCATION
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Benny', 'Jon', 'Justin', 'Tudor', 'Maor', "Ha-mil"]
# Initialize and start realtime video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set video width
cap.set(4, 480)  # set video height
# Define min window size to be recognized as a face
minW = 0.1 * cap.get(3)
minH = 0.1 * cap.get(4)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=Constants.SCALE_FACTOR,
        minNeighbors=Constants.MINIMUM_NEIGHBORS,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', frame)
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == ord('q'):
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()
