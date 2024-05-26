import cv2
from collections import defaultdict
from ultralytics import YOLO
import os
import shutil


# This function takes people's ids and their track history in xyxy format and crops then saves it

def get_track_history_and_crop_people(video_path):
    video_name = os.path.basename(video_path)
    video_name_without_extension = os.path.splitext(video_name)[0]

    print(f"Tracking and cropping for {video_name}")
    model = YOLO("../Models/yolov8n-seg.pt")

    cap = cv2.VideoCapture(video_path)

    # Store the track history
    track_history = defaultdict(lambda: [])

    cropped_people_path = f"cropped_people_{video_name_without_extension}"
    if os.path.exists(cropped_people_path):
        shutil.rmtree(cropped_people_path)
        os.mkdir(cropped_people_path)
    else:
        os.mkdir(cropped_people_path)

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
                    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    track.append((float(x), float(y), float(x2), float(y2), frame_index))  # x, x2, y, y2 points

                    crop_obj = frame[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                    # masking
                    object_name = f"{track_id}_{frame_index}"
                    cv2.imwrite(os.path.join(cropped_people_path, object_name + ".png"), crop_obj)

        else:
            # Break the loop if the end of the video is reached
            break

    print(f"Track history and cropped images folder successfully made for the video {video_name}")
    # Release the video capture object and close the display window
    cap.release()
    return track_history, cropped_people_path


# Testing function
# try_track_history, folder_name = get_track_history_and_crop_people("../Tests/office-1.mov")
