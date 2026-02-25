import cv2
import os
import imageio

RAW_DIR = "raw_data"
FRAME_DIR = "frames/train"
IMG_SIZE = 150
FRAMES_PER_FILE = 10   # extract 10 frames per gif/video

os.makedirs(FRAME_DIR, exist_ok=True)

def extract_from_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while cap.isOpened() and saved < FRAMES_PER_FILE:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(f"{output_dir}/{saved}.jpg", frame)
        saved += 1
    cap.release()

def extract_from_gif(gif_path, output_dir):
    gif = imageio.mimread(gif_path)
    for i, frame in enumerate(gif[:FRAMES_PER_FILE]):
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(f"{output_dir}/{i}.jpg", frame)

for label in ["bullying", "non_bullying"]:
    input_dir = os.path.join(RAW_DIR, label)
    output_dir = os.path.join(FRAME_DIR, label)
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        path = os.path.join(input_dir, file)
        if file.endswith(".mp4"):
            extract_from_video(path, output_dir)
        elif file.endswith(".gif"):
            extract_from_gif(path, output_dir)

print("✅ Frame extraction completed")
