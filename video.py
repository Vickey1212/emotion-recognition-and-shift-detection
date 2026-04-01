import os
import pandas as pd

video_folder = "D:/multimodal_emotion_recognition/CREMA-D/VideoFlash/"
output_csv = "D:/multimodal_emotion_recognition/video_emotion_labels.csv"

emotion_map = {
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "NEU": "neutral"
}

data = []
for file in os.listdir(video_folder):
    file_path = os.path.join(video_folder, file)
    parts = file.split("_")
    # Ensure filename has at least 3 parts and the emotion code is 3 characters
    if len(parts) >= 3 and len(parts[2]) == 3:
        emo_code = parts[2]
        if emo_code in emotion_map:
            label = emotion_map[emo_code]
            data.append({"filename": file, "emotion_label": label})
    else:
        print(f"Skipped file with unexpected format: {file}")
print(f"✅ Saved: {output_csv}")
df = pd.DataFrame(data, columns=["filename", "emotion_label"])
df.to_csv(output_csv, index=False)
print(f"Saved: {output_csv}")
