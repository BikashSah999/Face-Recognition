import cv2
import argparse
import numpy as np
import imutils
import os
import sys

ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--cascade", required=True, help="Path where the Face haar cascades reside")
ap.add_argument("-o", "--output", required=True, help="Path to save the dataset")
args = vars(ap.parse_args())

# detector = cv2.CascadeClassifier(args['cascade'])
detector = cv2.CascadeClassifier("face.xml")
image_num = 0
dataset_dir = "dataset/"
dir_in_dataset = []
cap = cv2.VideoCapture(0)
for x in os.listdir(dataset_dir):
    if os.path.isdir(os.path.join(dataset_dir, x)):
        dir_in_dataset.append(x)

if args['output'] in dir_in_dataset:
    print("Dataset Name Already Exist")
    sys.exit()
else:
    os.makedirs(dataset_dir+args['output'])

while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=400)
    image_org = np.copy(frame)
    image_gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if (len(rects)==1 and image_num<4):
        cv2.imwrite(dataset_dir+args["output"]+"/"+str(image_num)+".png", image_org[y-5:y+h+5, x-5:x+h+5])
        print(str(image_num)+" Image Captured")
        image_num += 1
    elif image_num==5:
        break

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
print(str(image_num)+" Total Images Captured")
