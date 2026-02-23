import cv2
import numpy as np
import tensorflow as tf
import face_recognition
from face_memory import FaceMemory
import time

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model("final_emotion_model.h5", compile=False)
class_names = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
IMG_SIZE = 96

memory = FaceMemory()

# -----------------------------
# Camera setup
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Emotion + Face Memory", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Emotion + Face Memory", 900, 700)

# -----------------------------
# Performance controls
# -----------------------------
frame_skip = 5
frame_count = 0
face_data = []
fps = 0
start_time = time.time()

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize for fast face detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Run detection every N frames
    if frame_count % frame_skip == 0:

        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        face_data = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            # Scale back to original frame
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            face = frame[top:bottom, left:right]
            if face.size == 0:
                continue

            # Emotion prediction
            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face_input = np.expand_dims(face_resized / 255.0, axis=0)

            preds = model(face_input, training=False).numpy()
            emotion = class_names[np.argmax(preds)]

            # Face recognition
            name = memory.recognize(face_encoding)

            face_data.append((top, right, bottom, left, name, emotion, face_encoding))

    # -----------------------------
    # Draw results
    # -----------------------------
    for (top, right, bottom, left, name, emotion, face_encoding) in face_data:

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} - {emotion}",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

    # -----------------------------
    # FPS CALCULATION
    # -----------------------------
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    start_time = end_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2)

    # -----------------------------
    # Show frame
    # -----------------------------
    cv2.imshow("Emotion + Face Memory", frame)

    key = cv2.waitKey(1)

    # Save new face
    if key == ord('s'):
        if len(face_data) > 0:
            new_name = input("Enter name: ")
            memory.save_face(new_name, face_data[0][6])

    # Exit
    if key == 27 or key == ord('q'):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
