import cv2
import Constants

cap = cv2.VideoCapture(0)
cap.set(3, 640) # set video width
cap.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier(Constants.HAARCASCADE_LOCATION)

# For each person, enter one numeric face id
face_id = input('\n enter user id and press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

# Initialize individual sampling face count
count = 0

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, Constants.MINIMUM_NEIGHBORS)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('frame', frame)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == ord('q'):
        break
    elif count >= Constants.NUMBER_OF_DATA_SAMPLES: # Take 100 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()
exec(open("face_trainer.py").read());