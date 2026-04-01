import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm

# Paths
VIDEO_FOLDER = 'CREMA-D/VideoFlash/'
OUTPUT_FILE = 'outputs/video_features/video_features.csv'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Frame sampling rate
FRAME_SAMPLE_RATE = 1  # One frame per second

# Collect all features
all_features = []

# Process each video file
video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".flv")]

print(f"📼 Found {len(video_files)} videos. Extracting features...")

for video_file in tqdm(video_files):
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * FRAME_SAMPLE_RATE)
    frame_count = 0
    embeddings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            try:
                reps = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)
                if reps and isinstance(reps, list):
                    embeddings.append(reps[0]["embedding"])
            except Exception:
                pass

        frame_count += 1

    cap.release()

    # Average embedding for this video
    if embeddings:
        mean_embedding = np.mean(embeddings, axis=0)
        row = [video_file] + list(mean_embedding)
        all_features.append(row)

# Column names
columns = ["filename"] + [f"face_{i+1}" for i in range(128)]

# Save to CSV
df = pd.DataFrame(all_features, columns=columns)
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Video features saved to: {OUTPUT_FILE}")
