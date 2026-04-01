# train_video_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("D:\\multimodal_emotion_recognition\\enhanced_combined_features.csv")

# Keep only FER emotion features (first 7 columns) + target
emotion_features = data.iloc[:, :7]  # first 7 columns are FER emotion probabilities
labels = data['emotion']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(emotion_features, labels, test_size=0.2, random_state=42, stratify=labels)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "D:\\multimodal_emotion_recognition\\outputs\\models\\video_emotion_model.pkl")
print("✅ Video-only emotion model saved successfully!")
