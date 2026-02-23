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
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5

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
# Load Pretrained Model
# ==============================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze entire base model first
base_model.trainable = False

# ==============================
# Build Model
# ==============================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

# ==============================
# Compile (Initial Training)
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining top layers first...\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS
)

# ==============================
# Fine-Tuning
# ==============================
print("\nStarting Fine-Tuning...\n")

# Unfreeze base model
base_model.trainable = True

# Freeze first 100 layers
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS
)

# ==============================
# Save Final Model
# ==============================
model.save("emotion_model_finetuned.h5")

print("\nFine-tuning complete. Model saved as emotion_model_finetuned.h5")
