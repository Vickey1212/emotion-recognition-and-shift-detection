import pandas as pd
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
num_samples = 1600  # total samples in your dataset
np.random.seed(42)

# 40 audio MFCC features
mfcc_features = [f"mfcc_{i}" for i in range(1, 41)]
# 12 chroma features
chroma_features = [f"chroma_{i}" for i in range(1, 13)]
# 20 mel features
mel_features = [f"mel_{i}" for i in range(1, 21)]
# 7 emotion probability features from video
video_features = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Combine all feature names
columns = mfcc_features + chroma_features + mel_features + video_features + ["emotion"]

# -----------------------------
# GENERATE RANDOMIZED FEATURES
# -----------------------------
data = np.random.rand(num_samples, len(columns) - 1)
emotions = np.random.choice(
    ["happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"],
    num_samples
)

# Combine features + labels
df = pd.DataFrame(data, columns=columns[:-1])
df["emotion"] = emotions

# -----------------------------
# SAVE TO FILE
# -----------------------------
output_path = r"D:\multimodal_emotion_recognition\features\combined_features.csv"
df.to_csv(output_path, index=False)

print(f"✅ combined_features.csv saved successfully at: {output_path}")
print(f"Shape: {df.shape}")
print(df.head())
