"""
That code uses a personal video to show face recognition using OpenCV and HAAR
To stop the program, press Q
"""

# Import opencv
import cv2

# Capture video file
# For webcam capture, enter: "0" or "1"
# For droidcam capture, enter something like http://localhost:8080/video
video_file = "../resources/face_recog.mp4"
video = cv2.VideoCapture(video_file)

# Read cascade classifier, that is going to classify something as a face
cascade_file = "../resources/haarcascade-frontalface-default.xml"
face_cascade = cv2.CascadeClassifier(cascade_file)

# Create infinite loop
while(True):

    # Get frame
    ret, frame = video.read()

    # Create a window named "Video" and resize it
    cv2.namedWindow("Video", 0)
    cv2.resizeWindow("Video", 288, 512)

    # Create gray image/frame, we need it to work on recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Here we collect all the faces found on the image/frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Now we loop through it getting X,Y, width and height of our face "boxes"
    for (x,y,w,h) in faces:
        #Then we draw the boxes around the faces
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3)

    # Here we show the image/frame on the "Video" window
    cv2.imshow("Video", frame)

    # Check if stop key was pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Free video cache
video.release()

# Destroy all open windows
cv2.destroyAllWindows()



