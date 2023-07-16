import cv2 #importing openCV module

faceRecognizer = cv2.face.LBPHFaceRecognizer_create() #Local Binary Patterns Histogram algo used to recognize faces
faceRecognizer.read('trainer.yml') #Loading the trained model
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #using Haar cascade algo to detect face from image

faceIDs = [0,1,2,3,4]

print("\n[INFO] Opening Camera...")

camera = cv2.VideoCapture(0)  #returning video from webcam
camera.set(3, 640) #setting frame width
camera.set(4, 480) #setting frame height

while(True):
    
    ret, img = camera.read() #storing data from camera in img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting color images to grayscale
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.05, #specifying how much the image size is reduced at each image scale
        minNeighbors = 4, #specifying how many neighbors each candidate rectangle should have to retain. Higher value(>5) results in less detections but with higher quality
        minSize = ((int) (0.1*camera.get(3)), (int) (0.1*camera.get(4))) #minimum image size. Images smaller than this are ignored
        )
    
    accuracy = 0 #initialising accuracy to 0
    
    for (x,y,w,h) in faces:
        faceID, confidence = faceRecognizer.predict(gray[y:y+h,x:x+w]) #predicting the face ID using the previously trained model
        #confidence = 0 is perfect match. Higher the number, lesser the confidence
        accuracy = 100-confidence #accuracy % = 100-confidence (general formula)
        
        if (accuracy >= 30 and accuracy <= 100): #if accuracy is high, display detected face ID and accuracy with green box
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            strFaceID = "Face ID:{0}".format(faceIDs[faceID])
            strAccuracy = "Accuracy:{0}%".format(round(accuracy))
        
        else: #if accuracy is low, display "Unknown" with red box
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
            strFaceID = "Unknown"
            strAccuracy = "Accuracy:{0}%".format(round(accuracy))
        
        #Displaying face ID as text
        cv2.putText(
            img,
            str(strFaceID),
            (x+5,y-5), #text position
            cv2.FONT_HERSHEY_COMPLEX_SMALL, #font
            1, (255,255,255), 1 #white colour with thickness=1
            )
        
        #Displaying accuracy as text
        cv2.putText(
            img,
            str(strAccuracy),
            (x+5,y+h+17), #text position
            cv2.FONT_HERSHEY_COMPLEX_SMALL, #font
            1, (255,0,0), 1 #blue colour with thickness=1
            ) 
    
    cv2.imshow('Camera', img) #display image

    waitKey = cv2.waitKey(1) & 0xff
    if (waitKey == 27) or (accuracy >= 60): #exit if ESC key is pressed or if accuracy>=60% --> face recognized succesfully
        break

if(accuracy >=60):
    print("\n[INFO] Face recognized with " + str(strFaceID) + "\n")
else:
    print("\n[INFO] Face not recognized. Please scan again!\n")

camera.release() #close camera
cv2.destroyAllWindows() #close all windows