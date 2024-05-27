import cv2
from collections import defaultdict
from ultralytics import YOLO
import os


# This function takes people's ids and their track history in xyxy format

def get_track_history(video_path):
    video_name = os.path.basename(video_path)

    print(f"Getting track history for {video_name}")
    model = YOLO("../../Models/yolov8n-seg.pt")

    cap = cv2.VideoCapture(video_path)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames on only frames
            results = model.track(frame, persist=True, classes=[0])  # classes=[0] only takes people

            if results[0].boxes.data.any():
                # Get the boxes and track IDs
                boxes = results[0].boxes.xyxy.cpu()
                if results[0].boxes.id is None:
                    continue

                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, x2, y2 = box
                    track = track_history[track_id]
                    track.append((float(x), float(y), float(x2), float(y2)))  # x, x2, y, y2 points

        else:
            # Break the loop if the end of the video is reached
            break

    print(f"Track history successfully made for the video {video_name}")
    # Release the video capture object and close the display window
    cap.release()
    return track_history


# Testing function
# try_track_history = get_track_history("../Tests/office-1.mov")
