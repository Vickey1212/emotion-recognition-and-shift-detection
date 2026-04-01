import os
import pandas as pd

# Path to your video files
VIDEO_FOLDER = "D:/multimodal_emotion_recognition/CREMA-D/VideoFlash"
OUTPUT_CSV = "D:/multimodal_emotion_recognition/video_emotion_labels.csv"

# Extract emotion label from filename
def extract_emotion_label(filename):
    emotion_code = int(filename.split('_')[2])
    emotion_map = {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fearful',
        7: 'disgust',
        8: 'surprised'
    }
    return emotion_map.get(emotion_code, "unknown")

# Generate label data
label_data = []
for file in os.listdir(VIDEO_FOLDER):
    if file.endswith(".flv"):
        emotion_label = extract_emotion_label(file)
        label_data.append({"filename": file, "emotion_label": emotion_label})

# Save to CSV
df = pd.DataFrame(label_data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved: {OUTPUT_CSV}")
