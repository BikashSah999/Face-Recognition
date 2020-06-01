import cv2
import argparse
import numpy as np
import imutils
import os
import sys

# "Command Line"
ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--cascade", required=True, help="Path where the Face haar cascades reside")
ap.add_argument("-o", "--output", required=True, help="Path to save the dataset")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier("face.xml")
image_num = 0
dataset_dir = "dataset/"
array_dir = "array/"
file_in_array = []

# Appending all files of dataset_dir in file_in_dataset
for x in os.listdir(array_dir):
    if os.path.isfile(os.path.join(array_dir, x)):
        file_in_array.append(x)

# Creating array name.npy and loading it to save name
if 'name.npy' in file_in_array:
    name_arr = np.load(array_dir+'name.npy')
    if args['output'] in name_arr:
        print("Name Already Exist")
        sys.exit()
    else:
        name_arr = np.append(name_arr, args['output'])
        np.save(array_dir+'name', name_arr)
else:
    name_arr = np.array([])
    name_arr = np.append(name_arr, args['output'])
    np.save(array_dir+'name', name_arr)

# Loading the index of the name
name_index = np.where(name_arr == args['output'])[0][0]

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=400)
    image_org = np.copy(frame)
    image_gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if len(rects) > 0:
        if key == ord("c"):
            cv2.imwrite(dataset_dir+args["output"]+"."+str(name_index)+"."+str(image_num)+".png", image_org[y-5:y+h+5, x-5:x+h+5])
            print(str(image_num)+" Image Captured")
            image_num += 1
        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print(str(image_num)+" Total Images Captured")
