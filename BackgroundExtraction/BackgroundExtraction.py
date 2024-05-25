import cv2
from ultralytics import YOLO
import os


# This function returns background and also saves it

def extract_background(video_path):
    video_name = os.path.basename(video_path)

    print(f"Extracting background from video {video_name} ...")
    model = YOLO("../Models/yolov8n-seg.pt")

    cap = cv2.VideoCapture(video_path)

    back_ground = "background.png"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model.predict(frame, show=False, classes=[0])  # classes=[0] takes only people

        if not results[0].boxes.data.any():
            if not os.path.exists(back_ground):
                back_ground_name = video_name + "_background.png"
                cv2.imwrite(back_ground_name, frame)
                print(f"People with no frame successfully found and saved as {back_ground_name} .")
                break

            continue

    # Release everything when done
    cap.release()
    return back_ground

# Testing function
# extract_background("../Tests/office-1.mov")
