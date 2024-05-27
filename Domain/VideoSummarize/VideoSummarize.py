import cv2
from ultralytics import YOLO
import os


# This function summarizes video with removing frames with no people in it

def summarize_video(video_path):
    video_name = os.path.basename(video_path)

    print(f"Summarizing video {video_name} ...")
    model = YOLO("../../Models/yolov8n-seg.pt")

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    summarized_video_name = f"{video_name}_summarized.avi"
    summarized_video_path = fr"C:\Users\ashas\PycharmProjects\video-synopsis-ai\Results\{summarized_video_name}"

    video_writer = cv2.VideoWriter(summarized_video_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps,
                                   (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model.predict(frame, show=False, classes=[0])  # classes=[0] takes only people

        if not results[0].boxes.data.any():
            print("Frame without people skipped.")
            continue

        video_writer.write(frame)

    print(f"Summarized video successfully written as {summarized_video_name}")
    # Release everything when done
    cap.release()
    video_writer.release()
    return summarized_video_path


# Testing function
# summarize_video("../Tests/office-1.mov")
