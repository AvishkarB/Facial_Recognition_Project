#importing required modules
import cv2
import numpy as np
from PIL import Image
import os

path = 'Dataset' #Path for database of all images

faceRecognizer = cv2.face.LBPHFaceRecognizer_create() #Local Binary Patterns Histogram algo used to recognize faces because of its computational simplicity
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #using Haar cascade algo to detect face from image

#function to get images as well as label data
def getImagesAndLabels(path):
    
    imagePaths = [os.path.join(path,file) for file in os.listdir(path)] #storing image paths in array
    faceSamples = []
    faceIDs = []
    
    for imagePath in imagePaths:
        PILImg = Image.open(imagePath).convert('L') #converting image to black and white
        numpyImg = np.array(PILImg, 'uint8') #storing image data in a numpy array
        
        faceID = int(os.path.split(imagePath)[-1].split(".")[1]) #retrieving face ID from image path
        faces = faceDetector.detectMultiScale(numpyImg) #detecting faces from image using Haar cascade
        
        for (x,y,w,h) in faces:
            faceSamples.append(numpyImg[y:y+h,x:x+w]) #storing detected face samples
            faceIDs.append(faceID) #labelling face samples (storing face ID for respective face sample)
        
    return faceSamples, faceIDs

print("\n[INFO] Training the model with captured faces. Please wait...")
faceSamples, faceIDs = getImagesAndLabels(path)
#Training our model with LBPH algo
faceRecognizer.train(faceSamples, np.array(faceIDs))

#Saving the model in .yml file
faceRecognizer.write('trainer.yml')

#Display the number of faces trained
print("\n[INFO] {0} Faces trained.\n".format(len(np.unique(faceIDs))))