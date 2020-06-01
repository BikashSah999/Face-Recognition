import cv2
import numpy as np
from PIL import Image
import os
# Path for face image database
path = 'dataset/'
array_path = 'array/'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("face.xml");

name_arr = np.load(array_path+'name.npy')

# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = []
    for x in os.listdir(path):
        imagePaths.append(os.path.join(path, x))
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        img_numpy = cv2.imread(imagePath, 0)
        id_index = imagePath.index('.') + 1
        id = int(imagePath[id_index])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples, ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
#print(np.array(ids))
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml')
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))