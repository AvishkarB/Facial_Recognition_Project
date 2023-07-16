# Face_Recognition_Project

What?
 + Created a **Face Detection and Recognition Security System** that can be used by IITH at the main gate to keep track of students.
 + It can serve as a replacement for the current ID card scanning technology that is being used. The current system is easily cheated by students carrying their friend's ID cards or generating a bar code online for their roll number on their phones and scanning their phones.
 + The advantage of my system is that it uses the face to identify the person, hence, there is no chance of getting cheated. Also, there is no requirement to carry an ID card at all times, which can get lost at times.
 + My system can also be used to mark student attendance as it allows multiple faces to be recognized at the same time.




How?
 + My project makes use of OpenCV library in Python for Computer Vision.
 + It is a Deep Learning model that employs Haar Cascade algorithm and LBPH (Local Binary Pattern Histogram) algorithm to detect and recognize the face.
 + Haar Cascade Algorithm - It is an algorithm that can detect objects in images, irrespective of their scale in image and location. This algorithm is not so complex and can run in real time. We can train a haar-cascade detector to detect various objects like cars, bikes, faces, buildings, fruits, etc. Haar cascade uses the cascading window, which tries to compute features in every window and classify whether it could be an object.
 > ![Alt text](https://miro.medium.com/v2/resize:fit:750/format:webp/1*XX8WqHo0lyrgZfTTRQ3ESQ.jpeg "Haar Cascade")
 + LBPH (Local Binary Pattern Histogram) Algorithm - It is a face-recognition algorithm known for its performance and how it is able to recognize the face of a person from both front face and side face. Using LBPH, we can represent the face images in the form of a simple data vector.
 > ![Alt text](https://editor.analyticsvidhya.com/uploads/658641%20J16_DKuSrnAH3WDdqwKeNA.png "LBPH")
 + We then use a Cascade Classifier, which is a machine learning-based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.




Running the program-
 + Save all the Python, .XML and .YML files in a folder.
 + Create a sub-folder named "Dataset" in the same folder for storing all the face images.
 + First, run faceCapturing.py
 + Then, faceTraining.py
 + And then, faceRecognition.py
