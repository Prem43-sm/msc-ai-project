import os
import cv2

data_path = r"D:\Pro2\Data"
print("Path Exists:", os.path.exists(data_path))

classes = os.listdir(data_path)

print("Classes Found:", classes)
print("-" * 40)

for emotion in classes:
    folder_path = os.path.join(data_path, emotion)
    images = os.listdir(folder_path)

    print(f"{emotion} -> {len(images)} images")

    # Check one sample image
    sample_image_path = os.path.join(folder_path, images[0])
    img = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)

    print(f"Sample Shape: {img.shape}")
    print("-" * 40)
