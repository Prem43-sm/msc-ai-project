import os
import shutil
import random

input_path = r"D:\Pro1\Processed_Data"
output_path = r"D:\Pro1\Final_Dataset"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

classes = os.listdir(input_path)

for emotion in classes:
    emotion_path = os.path.join(input_path, emotion)
    images = os.listdir(emotion_path)

    random.shuffle(images)

    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    for dataset_type, dataset_images in zip(
        ["train", "val", "test"],
        [train_images, val_images, test_images]
    ):
        dataset_folder = os.path.join(output_path, dataset_type, emotion)
        os.makedirs(dataset_folder, exist_ok=True)

        for img in dataset_images:
            src = os.path.join(emotion_path, img)
            dst = os.path.join(dataset_folder, img)
            shutil.copy(src, dst)

    print(f"{emotion} split completed.")

print("Dataset splitting done successfully.")
