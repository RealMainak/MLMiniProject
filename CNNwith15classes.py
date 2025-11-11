# ============================================================
# 🧠 Plant Disease Classification (CNN - TensorFlow/Keras)
# Dataset: https://www.kaggle.com/datasets/emmarex/plantdisease
# Compatible with: TF 2.10.1 + CUDA 11.x (MX130 GPU)
# ============================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import time
import numpy as np  # 👈 Added for class weights
from sklearn.utils import class_weight  # 👈 Added for class weights

# ------------------------------------------------------------
# GPU configuration (optional)
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
# Paths
# ------------------------------------------------------------
DATASET_DIR = r"C:\Users\bosem\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\plantvillage"  # 👈 change this to your extracted dataset path

# Check data structure
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError("❌ Dataset folder not found! Please set correct DATASET_DIR path.")

# ------------------------------------------------------------
# Data Preprocessing
# ------------------------------------------------------------
IMG_SIZE = (128, 128)  # Reduce to fit in 2GB VRAM (MX130)
BATCH_SIZE = 16        # Keep small batch size to prevent OOM

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
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
print(f"📚 Number of classes: {num_classes}")

# ------------------------------------------------------------
# NEW: Dataset Balancing (Calculate Class Weights)
# ------------------------------------------------------------

# Get the class indices for all training samples
training_classes = train_gen.classes

# Get the unique class labels
unique_classes = np.unique(training_classes)

# Calculate class weights using scikit-learn's 'balanced' mode
# This automatically gives more weight to minority classes
weights = class_weight.compute_class_weight(
    'balanced',
    classes=unique_classes,
    y=training_classes
)

# Create a dictionary mapping class index to its calculated weight
class_weights = dict(zip(unique_classes, weights))

# --- Display the calculated weights clearly ---
print("\n--- Class Weights ---")

# Create a reverse mapping from index -> class name
index_to_name_map = {index: name for name, index in train_gen.class_indices.items()}

# Find the longest class name for nice formatting
max_name_len = max(len(name) for name in train_gen.class_indices.keys())

# Iterate and print
for index, weight in class_weights.items():
    class_name = index_to_name_map[index]
    print(f"{class_name:<{max_name_len}} | Weight: {weight:.4f}")

print("---------------------\n")


# ------------------------------------------------------------
# Model Architecture
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
# Compile the Model
# ------------------------------------------------------------
# Setting a slightly lower learning rate (default is 0.001)
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 

model.compile(optimizer=custom_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ------------------------------------------------------------
# Train the Model
# ------------------------------------------------------------
EPOCHS = 30

start = time.time()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,  # 👈 Apply the calculated weights
    verbose=1
)
end = time.time()
print(f"\n⏱ Training completed in {(end - start):.2f} seconds")

# ------------------------------------------------------------
# Evaluate the Model
# ------------------------------------------------------------
loss, acc = model.evaluate(val_gen, verbose=0)
print(f"✅ Validation Accuracy: {acc*100:.2f}%")

# ------------------------------------------------------------
# Save Model
# ------------------------------------------------------------
model.save("plant_disease_cnn_model.h5")
print("💾 Model saved as 'plant_disease_cnn_model.h5'")

# ------------------------------------------------------------
# Plot Results
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
plt.show()
