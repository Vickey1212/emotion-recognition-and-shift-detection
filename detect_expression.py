# # detect_expression.py
# import cv2
# import numpy as np
# import joblib
# from fer import FER

# # Load model
# model = joblib.load("D:/multimodal_emotion_recognition/outputs/models/video_emotion_model.pkl")
# print("Video-only model loaded successfully!")

# # Initialize FER detector without MTCNN
# detector = FER(mtcnn=False)

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = detector.detect_emotions(rgb_frame)

#     if results:
#         (x, y, w, h) = results[0]["box"]
#         emotions = results[0]["emotions"]

#         # Convert emotion dictionary to feature vector
#         features = np.array(list(emotions.values())).reshape(1, -1)

#         # Predict emotion using your model
#         predicted_emotion = model.predict(features)[0]

#         # Draw results on frame
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, predicted_emotion, (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

#     cv2.imshow(" Real-Time Emotion Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Its working YAY...")



# detect_expression.py
import cv2
import numpy as np
import joblib
import time # <--- NEW IMPORT
from fer import FER

# --- INITIAL SETUP ---
# Load model
# NOTE: Ensure the path "D:/multimodal_emotion_recognition/outputs/models/video_emotion_model.pkl" is correct on your system.
model = joblib.load("D:/multimodal_emotion_recognition/outputs/models/video_emotion_model.pkl")
print("Video-only model loaded successfully!")

# Initialize FER detector without MTCNN
detector = FER(mtcnn=False)

# Start webcam
cap = cv2.VideoCapture(0)

# --- TIMER VARIABLES ---
INTERVAL = 5.0  # Capture and process every 5.0 seconds
last_capture_time = time.time()
# Variables to store the last successful prediction and box coordinates
current_emotion = "Initializing..."
current_box = None
# --- END TIMER VARIABLES ---


while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    
    # --- CHECK IF IT'S TIME TO CAPTURE ---
    if current_time - last_capture_time >= INTERVAL:
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. DETECT EMOTIONS on the sampled frame
        results = detector.detect_emotions(rgb_frame)

        if results:
            # Update the box and emotion for display
            current_box = results[0]["box"]
            emotions = results[0]["emotions"]

            # Convert emotion dictionary to feature vector
            features = np.array(list(emotions.values())).reshape(1, -1)

            # 2. PREDICT emotion using your model
            predicted_emotion = model.predict(features)[0]
            current_emotion = predicted_emotion
        else:
            # If no face is detected in the 5-second sample
            current_emotion = "Face not detected"
            current_box = None

        # Reset the timer to the current time
        last_capture_time = current_time

    # --- DISPLAY RESULTS (Runs every frame) ---
    if current_box is not None:
        (x, y, w, h) = current_box
        
        # Draw the rectangle using the last captured box coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display the last predicted emotion
        cv2.putText(frame, current_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    else:
        # Display a status message if no prediction/face is available
        cv2.putText(frame, current_emotion, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
    cv2.imshow(" Real-Time Emotion Detection (5s Interval)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Its working YAY... (5s sampling complete)")
