import os
import cv2

input_path = r"D:\Pro1\Data"
output_path = r"D:\Pro1\Processed_Data"

IMG_SIZE = 96

classes = os.listdir(input_path)

for emotion in classes:
    input_emotion_path = os.path.join(input_path, emotion)
    output_emotion_path = os.path.join(output_path, emotion)

    os.makedirs(output_emotion_path, exist_ok=True)

    images = os.listdir(input_emotion_path)

    print(f"Processing {emotion}...")

    for img_name in images:
        img_path = os.path.join(input_emotion_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        # Resize image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Save processed image
        save_path = os.path.join(output_emotion_path, img_name)
        cv2.imwrite(save_path, img)

    print(f"{emotion} Done.")
    print("-" * 30)

print("All images processed successfully.")
