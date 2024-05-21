import cv2
import numpy as np

# Load the video
video_path = '../Tests/denemevideo.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# Read the first frame
ret, frame = cap.read()

# Check if the frame was read successfully
if not ret:
    print("Error: Unable to read the frame.")
    exit()

# Save the first frame as an image
cv2.imwrite('first_frame.jpg', frame)

# Release the video capture object
cap.release()

# remove people from first frame background
