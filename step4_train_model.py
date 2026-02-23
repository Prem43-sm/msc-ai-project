import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================
# Paths
# ==============================
train_path = r"D:\Pro1\Final_Dataset\train"
val_path = r"D:\Pro1\Final_Dataset\val"

IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 10

# ==============================
# Data Generators
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ==============================
# Load MobileNetV2
# ==============================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Freeze base model

# ==============================
# Custom Head
# ==============================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

# ==============================
# Compile
# ==============================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# Train
# ==============================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ==============================
# Save Model
# ==============================
model.save("emotion_model.h5")

print("Model training complete and saved.")
