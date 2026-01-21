import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
import os
import sys
import matplotlib.pyplot as plt

# ===========================================
# CONFIGURATION
# ===========================================
# 1. CHOOSE YOUR INPUT DATASET
# Options: 'STANDARD' (Original/No Dropout) or 'MC_DROPOUT' (Uncertainty-based)
DATASET_MODE = 'STANDARD'  # Change to 'MC_DROPOUT' as needed

# 2. Define Base Paths
# Update these specific paths to match where your two different curriculum folders are located
if DATASET_MODE == 'STANDARD':
    print("🔵 CONFIGURATION: Loading STANDARD Curriculum (No MC Dropout)")
    CURRICULUM_DIR = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\StudentModel1\plantvillage_curriculum_standard"
    
elif DATASET_MODE == 'MC_DROPOUT':
    print("🟣 CONFIGURATION: Loading MC DROPOUT Curriculum (Uncertainty-based)")
    # This should match the TARGET_DATA_DIR from your MC Dropout generator script
    CURRICULUM_DIR = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\StudentModel1\plantvillage_curriculum_mcdropout"
    
else:
    raise ValueError("Invalid DATASET_MODE. Choose 'STANDARD' or 'MC_DROPOUT'.")

# 3. Output Directory
SAVE_DIR = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\StudentModel1"

# Ensure save directory exists
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"✅ Created save directory: {SAVE_DIR}")

# Image settings (PDF suggests 96x96 or 128x128 for TinyML)
IMG_SIZE = (128, 128) 
BATCH_SIZE = 32
NUM_CLASSES = 15

# Training Hyperparameters
EPOCHS_STAGE_1 = 20  # Easy data
EPOCHS_STAGE_2 = 15  # Medium data
EPOCHS_STAGE_3 = 15  # Hard data

# ===========================================
# 1. DEFINE ENHANCED TINY-ML MODEL
# ===========================================
def build_tiny_student(input_shape, num_classes):
    print("Building ENHANCED Student Model (Target: ~1.5 - 2.0 MB)...")
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=input_shape),
        
        # Block 1: Increased filters to 32 (was 16)
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2: Increased filters to 64 (was 32)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3: Increased filters to 128 (was 64)
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4: Deep features
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 5: NEW EXTRA BLOCK (Helps with 'Hard' samples)
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier Head 
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Intermediate Dense Layer for better decision boundaries
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ===========================================
# 2. HELPER: TRAIN ONE STAGE
# ===========================================
def train_stage(stage_name, train_dir, model, epochs, learning_rate, initial_epoch=0):
    print(f"\n{'='*40}")
    print(f"🚀 STARTING {stage_name} (LR={learning_rate})")
    print(f"   Source: {train_dir}")
    print(f"{'='*40}")

    if not os.path.exists(train_dir):
        print(f"❌ Error: Directory not found: {train_dir}")
        print("   Please check that your Generator script has finished and the path is correct.")
        sys.exit()

    # Data Generator (Augmentation helps generalization)
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', shuffle=True
    )
    
    val_gen = datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False
    )

    # Compile (Re-compile to update Learning Rate)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )
    return history, model

# ===========================================
# 3. MAIN PIPELINE
# ===========================================
# Initialize Model
print(f"--- Building Tiny Student Model (Mode: {DATASET_MODE}) ---")
student_model = build_tiny_student(IMG_SIZE + (3,), NUM_CLASSES)
student_model.summary()

# --- Check Model Size ---
total_params = student_model.count_params()
size_in_mb = (total_params * 4) / (1024 * 1024)
print(f"\n📦 Estimated Model Size: {size_in_mb:.2f} MB (Target < 2.5 MB)")

full_history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

# --- STAGE 1: EASY SAMPLES ---
dir_d1 = os.path.join(CURRICULUM_DIR, "D1_Easy")
hist1, student_model = train_stage("STAGE 1 (Easy)", dir_d1, student_model, 
                                   epochs=EPOCHS_STAGE_1, learning_rate=1e-3)

# Save checkpoint
stage1_filename = f"student_stage1_weights_{DATASET_MODE}.h5"
stage1_path = os.path.join(SAVE_DIR, stage1_filename)
student_model.save_weights(stage1_path)
print(f"✅ Stage 1 weights saved to: {stage1_path}")

# --- STAGE 2: MEDIUM SAMPLES ---
dir_d2 = os.path.join(CURRICULUM_DIR, "D2_Medium")
hist2, student_model = train_stage("STAGE 2 (Medium)", dir_d2, student_model, 
                                   epochs=EPOCHS_STAGE_2, learning_rate=1e-4)

# --- STAGE 3: HARD SAMPLES ---
dir_d3 = os.path.join(CURRICULUM_DIR, "D3_Hard")
hist3, student_model = train_stage("STAGE 3 (Hard)", dir_d3, student_model, 
                                   epochs=EPOCHS_STAGE_3, learning_rate=1e-5)

# ===========================================
# 4. FINAL SAVE & PLOT
# ===========================================
final_filename = f"student_final_curriculum_{DATASET_MODE}.h5"
final_model_path = os.path.join(SAVE_DIR, final_filename)
print(f"\n💾 Saving Final Student Model to: {final_model_path}")
student_model.save(final_model_path)

# Combine histories for plotting
for h in [hist1, hist2, hist3]:
    full_history['accuracy'].extend(h.history['accuracy'])
    full_history['val_accuracy'].extend(h.history['val_accuracy'])
    full_history['loss'].extend(h.history['loss'])
    full_history['val_loss'].extend(h.history['val_loss'])

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(full_history['accuracy'], label='Train Acc')
plt.plot(full_history['val_accuracy'], label='Val Acc')
# Add vertical lines to show stage transitions
stage1_end = EPOCHS_STAGE_1
stage2_end = EPOCHS_STAGE_1 + EPOCHS_STAGE_2
plt.axvline(x=stage1_end, color='r', linestyle='--', label='End Stage 1')
plt.axvline(x=stage2_end, color='g', linestyle='--', label='End Stage 2')
plt.title(f'Curriculum Accuracy ({DATASET_MODE})')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(full_history['loss'], label='Train Loss')
plt.plot(full_history['val_loss'], label='Val Loss')
plt.axvline(x=stage1_end, color='r', linestyle='--')
plt.axvline(x=stage2_end, color='g', linestyle='--')
plt.title(f'Curriculum Loss ({DATASET_MODE})')
plt.legend()

plt.tight_layout()
plt.show()