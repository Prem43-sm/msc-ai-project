import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# Paths
# ==============================
test_path = r"D:\Pro1\Final_Dataset\test"

IMG_SIZE = 96
BATCH_SIZE = 32

# ==============================
# Load Test Data
# ==============================
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ==============================
# Load Trained Model
# ==============================
# model = tf.keras.models.load_model("emotion_model.h5")
model = tf.keras.models.load_model("final_emotion_model.h5")

# ==============================
# Evaluate Accuracy
# ==============================
loss, accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# ==============================
# Predictions
# ==============================
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# ==============================
# Classification Report
# ==============================
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# ==============================
# Confusion Matrix
# ==============================
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
