"""
That code uses a Friends TV Show photo to show face recognition using OpenCV and HAAR
To stop the program, press any key
You can change the image by changing img_file, there, change friends.jpg to friends2.jpg
"""

# Import opencv
import cv2

# Importing png image
# It generates a matrix of pixels, with rows and columns
img_file = "../resources/friends.jpg"
img = cv2.imread(img_file)

# Read cascade classifier, that is going to classify something as a face
cascade_file = "../resources/haarcascade-frontalface-default.xml"
face_cascade = cv2.CascadeClassifier(cascade_file)

# Create gray image/frame, we need it to work on recognition
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

scaleFactor = 1.1
minNeighbors = 7
# Here we collect all the faces found on the image/frame
faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)

# Now we loop through it getting X,Y, width and height of our face "boxes"
for (x,y,w,h) in faces:
    #Then we draw the boxes around the faces
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)

# Create a new window and show the image/frame
cv2.imshow("Image", img)

# Wait for key press to close window
cv2.waitKey(0)