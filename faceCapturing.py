import cv2 #importing openCV module

camera = cv2.VideoCapture(0) #returning video from webcam
camera.set(3,640) #setting frame width
camera.set(4,480) #setting frame height

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #using Haar cascade algo to detect face from image

faceID = input("\nEnter Face ID: ") #Enter number between 0-4

print("\n[INFO] Initializing Face Capture...")

imgCount = 0 #counting the number of face samples
maxSamples = 50 #maximum number of face samples. Higher the no. of samples, greater the accuracy but more time consuming

while(True):
    
    ret, img = camera.read() #storing data from camera in img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting color images to grayscale
    faces = faceDetector.detectMultiScale(gray, 1.3, 5) #detecting faces from grayscale images
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        imgCount+=1
        
        #saving the captured images in a folder
        cv2.imwrite("Dataset/User." + str(faceID) + "." + str(imgCount) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('Capturing Face', img)

    waitKey = cv2.waitKey(25) & 0xff
    if (waitKey == 27) or (imgCount >= maxSamples): #exit if ESC key is pressed or if number of face samples >= maximum samples
        break

print("\n[INFO] Face Capture Complete.\n")
camera.release() #close camera
cv2.destroyAllWindows() #close all windows