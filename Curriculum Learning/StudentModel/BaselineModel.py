import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
import os
import sys
import matplotlib.pyplot as plt

# ===========================================
# CONFIGURATION
# ===========================================
# Path to the ORIGINAL balanced dataset (Mixed difficulty)
# (Using the path we verified in the previous steps)
TRAIN_DIR = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\TeacherModel\plantvillage_whole_balanced"
SAVE_DIR = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\StudentModel"

# Same settings as Curriculum Model for fair comparison
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 15
TOTAL_EPOCHS = 50  # Matches the total curriculum epochs (20 + 15 + 15)

# ===========================================
# 1. DEFINE EXACT SAME MODEL
# ===========================================
def build_enhanced_student(input_shape, num_classes):
    print("Building Baseline Student Model...")
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 5
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Head
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ===========================================
# 2. STANDARD TRAINING PIPELINE
# ===========================================
print(f"--- Starting Baseline Training on: {TRAIN_DIR} ---")

if not os.path.exists(TRAIN_DIR):
    print(f"❌ Error: Directory not found: {TRAIN_DIR}")
    sys.exit()

# Data Generator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', shuffle=True
)

val_gen = datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False
)

# Build & Compile
model = build_enhanced_student(IMG_SIZE + (3,), NUM_CLASSES)

# We use a standard Learning Rate schedule (ReduceLROnPlateau) 
# instead of the manual stage-based LR, as this is standard practice.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=TOTAL_EPOCHS,
    callbacks=[reduce_lr]
)

# ===========================================
# 3. SAVE & PLOT
# ===========================================
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

save_path = os.path.join(SAVE_DIR, "student_baseline_standard.h5")
print(f"\n💾 Saving Baseline Model to: {save_path}")
model.save(save_path)

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Baseline (Standard) Training Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Baseline (Standard) Training Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()