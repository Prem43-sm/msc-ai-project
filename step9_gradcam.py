import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load full model
model = tf.keras.models.load_model("best_emotion_model.h5")

# Split model
base_model = model.layers[0]   # MobileNetV2
head_model = tf.keras.Sequential(model.layers[1:])  # GAP + Dense layers

# Find last conv layer
last_conv_layer = None
for layer in base_model.layers[::-1]:
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break

print("Using layer:", last_conv_layer.name)

# Image settings
IMG_SIZE = 96
class_names = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']

img_path = r"C:\Users\Pc495\Downloads\smile.jpeg"

img = cv2.imread(img_path)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_array = np.expand_dims(img / 255.0, axis=0)

# Forward + Gradients
with tf.GradientTape() as tape:

    # Get feature maps from MobileNet
    feature_maps = base_model(img_array)

    tape.watch(feature_maps)

    # Get predictions from head
    preds = head_model(feature_maps)

    class_index = tf.argmax(preds[0])
    loss = preds[:, class_index]

# Compute gradients
grads = tape.gradient(loss, feature_maps)

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

feature_maps = feature_maps[0]

heatmap = feature_maps @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

# Convert to heatmap
heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Overlay
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# Show result
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title(class_names[class_index])
plt.axis("off")
plt.show()
