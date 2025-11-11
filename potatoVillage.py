# ============================================================
# 🌿 Potato Leaf Disease Classification (3 Classes)
# Dataset: https://www.kaggle.com/datasets/emmarex/plantdisease
# Classes: Potato___Early_blight, Potato___Late_blight, Potato___healthy
# ============================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# ------------------------------------------------------------
# GPU Configuration
# ------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU detected:", gpus)
    except Exception as e:
        print("⚠️ GPU setup issue:", e)
else:
    print("⚠️ No GPU found. Running on CPU.")

# ------------------------------------------------------------
# Dataset Directory (CHANGE THIS)
# ------------------------------------------------------------
DATASET_DIR = r"C:\Users\bosem\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\plantvillage"  # 👈 change this to your extracted dataset path

# ------------------------------------------------------------
# Select only the required 3 classes
# ------------------------------------------------------------
TARGET_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

# Create a new directory for filtered data (optional)
FILTERED_DIR = os.path.join(os.path.dirname(DATASET_DIR), "PlantVillage_Potato_Only")
os.makedirs(FILTERED_DIR, exist_ok=True)

for cls in TARGET_CLASSES:
    src_folder = os.path.join(DATASET_DIR, cls)
    dst_folder = os.path.join(FILTERED_DIR, cls)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        # copy only if not already done
        for f in os.listdir(src_folder):
            src_path = os.path.join(src_folder, f)
            dst_path = os.path.join(dst_folder, f)
            if not os.path.exists(dst_path):
                try:
                    import shutil
                    shutil.copy(src_path, dst_path)
                except:
                    pass

DATASET_DIR = FILTERED_DIR  # Now we’ll use the filtered dataset

# ------------------------------------------------------------
# Data Preprocessing
# ------------------------------------------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_gen.class_indices)
print(f"\n📚 Classes found: {num_classes}")
for i, (cls, idx) in enumerate(train_gen.class_indices.items()):
    print(f"{i+1:02d}. {cls} (index: {idx})")

# ------------------------------------------------------------
# Display number of images per class
# ------------------------------------------------------------
print("\n📂 Image counts per class:")
class_counts = {}
for cls, idx in train_gen.class_indices.items():
    folder = os.path.join(DATASET_DIR, cls)
    count = len(os.listdir(folder))
    class_counts[cls] = count
    print(f"{idx:02d}. {cls:30s} -> {count:4d} images")

total_images = sum(class_counts.values())
print(f"\n📊 Total images: {total_images}\n")

# ------------------------------------------------------------
# Build the CNN Model
# ------------------------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# ------------------------------------------------------------
# Compile and Train
# ------------------------------------------------------------
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 8

start = time.time()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1
)
end = time.time()
print(f"\n⏱ Training completed in {(end - start):.2f} seconds")

# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------
loss, acc = model.evaluate(val_gen, verbose=0)
print(f"✅ Validation Accuracy: {acc*100:.2f}%")

# ------------------------------------------------------------
# Save Model
# ------------------------------------------------------------
model.save("potato_disease_cnn_model.h5")
print("💾 Model saved as 'potato_disease_cnn_model.h5'")

# ------------------------------------------------------------
# Plot Accuracy and Loss
# ------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.tight_layout()
plt.show()
