import cv2
import numpy as np
import tensorflow as tf
import face_recognition
import pandas as pd
import os
from datetime import datetime
from face_memory import FaceMemory

# -----------------------------
# Load emotion model
# -----------------------------
model = tf.keras.models.load_model("best_emotion_model.h5", compile=False)
class_names = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
IMG_SIZE = 96

memory = FaceMemory()

# -----------------------------
# Attendance folder
# -----------------------------
ATTENDANCE_DIR = "attendance"
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

today_date = datetime.now().strftime("%Y-%m-%d")
csv_file = os.path.join(ATTENDANCE_DIR, f"{today_date}.csv")

# -----------------------------
# Create file if not exists
# -----------------------------
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["Name", "Time", "Emotion"])
    df.to_csv(csv_file, index=False)

# -----------------------------
# Load today's attendance
# -----------------------------
df = pd.read_csv(csv_file)
marked_names = set(df["Name"].values)

# -----------------------------
# Start camera
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

process_this_frame = True
face_data = []

cv2.namedWindow("AI Attendance System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Attendance System", 900, 700)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        face_data = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            name = memory.recognize(face_encoding)

            face = frame[top:bottom, left:right]
            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face_normalized = face_resized / 255.0
            face_input = np.reshape(face_normalized, (1, IMG_SIZE, IMG_SIZE, 3))

            preds = model.predict(face_input, verbose=0)
            emotion = class_names[np.argmax(preds)]

            # -----------------------------
            # Mark attendance only once
            # -----------------------------
            if name != "Unknown" and name not in marked_names:

                time_now = datetime.now().strftime("%H:%M:%S")

                new_entry = pd.DataFrame([[name, time_now, emotion]],
                                         columns=["Name", "Time", "Emotion"])

                new_entry.to_csv(csv_file, mode='a', header=False, index=False)

                marked_names.add(name)

                print(f"{name} marked present at {time_now} with {emotion}")

            face_data.append((top, right, bottom, left, name, emotion))

    process_this_frame = not process_this_frame

    # -----------------------------
    # Draw on screen
    # -----------------------------
    for (top, right, bottom, left, name, emotion) in face_data:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} - {emotion}", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("AI Attendance System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
