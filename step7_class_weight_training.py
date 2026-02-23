import tensorflow as tf

# ==============================
# ✅ GPU SETTINGS
# ==============================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# ==============================
# Imports
# ==============================
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ==============================
# Paths
# ==============================
train_path = r"D:\Pro1\Final_Dataset\train"
val_path   = r"D:\Pro1\Final_Dataset\val"

IMG_SIZE = 96
BATCH_SIZE = 64   # 🔥 GPU can handle bigger batch
EPOCHS = 35

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
# Compute Class Weights
# ==============================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weight_dict = dict(enumerate(class_weights))

print("\nClass Weights:")
print(class_weight_dict)

# ==============================
# Load Pretrained MobileNetV2
# ==============================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = True

# Freeze first layers
for layer in base_model.layers[:100]:
    layer.trainable = False

# ==============================
# Build Model
# ==============================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(5, activation='softmax', dtype='float32')  # ⚠ important
])

# ==============================
# Compile
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# Callbacks
# ==============================
callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "best_emotion_model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-6
    )
]

# ==============================
# Train
# ==============================
history = model.fit(

    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# ==============================
# Save Final Model
# ==============================
model.save("final_emotion_model.h5")

print("\n✅ TRAINING COMPLETE")
