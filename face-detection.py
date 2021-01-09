import cv2 
from random import randrange

# Load some pre-trained data on face frontals from openCV

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

# Iterate forever over frames 
while True: 
    # Read current frame 
    successful_frame_read, frame = webcam.read() 
    # Convert to greyscale 
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces 
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, x+h), (randrange(128,256), randrange(128,256), randrange(128,255)), 10)

    cv2.imshow('Clever Programmer Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key==ord('q') or key==ord('Q'):
        break

# Release the VideoCapture object
webcam.release() 

"""
# Choose an image to detect faces in 
#img = cv2.imread('photo3x4.jpg')

# Convert to greyscale 
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces 
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, x+h), (randrange(128,256), randrange(128,256), randrange(128,255)), 10)

# Display the image with the faces and rectangle drawn 
cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey()
"""

print('Code completed')