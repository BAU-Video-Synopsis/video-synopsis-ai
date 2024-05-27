import os
import cv2
from Domain.BackgroundExtraction.BackgroundExtraction import extract_background
from Domain.VideoSummarize.VideoSummarize import summarize_video
from Domain.ObjectTracking.ObjectTrackingv2 import get_track_history_and_crop_people


def video_synopsis(video_path):
    video_name = os.path.basename(video_path)
    video_name_without_extension = os.path.splitext(video_name)[0]
    synopsis_output_name = f"{video_name_without_extension}_output.avi"

    background_path = extract_background(video_path)
    background_image = cv2.imread(background_path)

    summarized_video_path = summarize_video(video_path)

    # track_history, cropped_people_folder_path = get_track_history_and_crop_people(summarized_video_path) summarized gidince bozuluyor
    track_history, cropped_people_folder_path = get_track_history_and_crop_people(video_path)

    cap = cv2.VideoCapture(summarized_video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(synopsis_output_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    max_locations = max(len(v) for v in track_history.values())

    for i in range(max_locations):
        copy_background = background_image.copy()
        for person_id, locations in track_history.items():
            # Check if location index exists for this person
            if i < len(locations):
                x, y, x2, y2, frame_index = locations[i]
                cropped_image_path = f"{cropped_people_folder_path}/{person_id}_{frame_index}.png"
                cropped_image = cv2.imread(cropped_image_path)
                copy_background[int(y):int(y2), int(x):int(x2)] = cropped_image
        video_writer.write(copy_background)

    cap.release()
    video_writer.release()
    return synopsis_output_name


# Testing function
# synopsis_output_path = video_synopsis("../Tests/office-5.mov")
