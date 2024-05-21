import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

# Define video capture and output
input_video_path = '../Tests/cctv.mp4'
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
video_writer = cv2.VideoWriter("summarized_video.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model.predict(frame, show=False, classes=[0], device='cuda')  # classes=[0] takes only people in cropping

    if not results[0].boxes.data.any():
        print("frame is skipped")
        continue

    video_writer.write(frame)


# Release everything when done
cap.release()
video_writer.release()
cv2.destroyAllWindows()
