import cv2
import numpy as np
import os
import imutils

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier("face.xml");
font = cv2.FONT_HERSHEY_SIMPLEX
# initiate id counter
id = 0

array_dir = "array/"
name_array = np.load(array_dir+'name.npy')

# Initialize and start realtime video capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # If confidence is less then 100 ==> "0" : perfect match
        if (confidence < 100):
            id = name_array[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(
            frame,
            str(id),
            (x + 5, y - 5),
            font,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            frame,
            str(confidence),
            (x + 5, y + h - 5),
            font,
            1,
            (255, 255, 0),
            1
        )

    cv2.imshow('camera', frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()