import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
# Import all models we'll need
from tensorflow.keras.applications import MobileNetV2, ResNet50V2
# Import their specific preprocessing functions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import sys
import shutil
import random

# ===========================================
# STEP 0: LOCAL SETUP
# ===========================================
print("TensorFlow Version:", tf.__version__)
print("Available GPU(s):", tf.config.list_physical_devices('GPU'))

# ===========================================
# USER SETTINGS & PATHS
# ===========================================
# --- !!! CHOOSE YOUR MODEL HERE !!! ---
MODEL_CHOICE = 'resnet'  # Options: 'mobilenet', 'resnet'
# ------------------------------------

# --- PATHS UPDATED FOR LOCAL PC ---
# !!! YOU MUST CHANGE THIS PATH !!!
# Set this to the full path of the 'PlantVillage' folder containing all 38 class folders.
original_dataset_dir = r"C:\Users\bosem\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\plantvillage" 

# MODIFIED: Changed folder name to reflect the whole dataset
balanced_dir = "plantvillage_whole_balanced" 
# -------------------------------

# MODIFIED: Removed hardcoded 'selected_classes' list.
# We will detect them dynamically in Step 1.

img_size = (224, 224)
batch_size = 32 # Increased batch size slightly as the dataset is larger
val_split = 0.2
images_per_class = 950 # Limits dominant classes to 950 images
stage_1_epochs = 10    # Slightly reduced initial epochs as dataset is larger
stage_2_epochs = 20
total_epochs = stage_1_epochs + stage_2_epochs
random.seed(42)

print(f"--- INITIALIZING WHOLE DATASET EXPERIMENT FOR: {MODEL_CHOICE} ---")

# ===========================================
# STEP 1 & 2: CREATE BALANCED DATASET (ALL CLASSES)
# ===========================================

# Quick check to ensure the user updated the path
if "CHANGE/ME" in original_dataset_dir:
    print("="*50)
    print("ERROR: Please update the 'original_dataset_dir' variable.")
    print("="*50)
    sys.exit()

# MODIFIED: Dynamically find all subdirectories (classes)
all_classes = sorted([d for d in os.listdir(original_dataset_dir) 
                      if os.path.isdir(os.path.join(original_dataset_dir, d))])
num_classes = len(all_classes)
print(f"Found {num_classes} classes in source directory.")

print(f"Checking for balanced dataset at: {balanced_dir}")
if not os.path.exists(balanced_dir):
    print(f"Creating balanced dataset at: {balanced_dir}")
    os.makedirs(balanced_dir, exist_ok=True)
    
    # Iterate over ALL detected classes
    for class_name in all_classes:
        src_class_dir = os.path.join(original_dataset_dir, class_name)
        dst_class_dir = os.path.join(balanced_dir, class_name)
        
        os.makedirs(dst_class_dir, exist_ok=True)
        
        all_images = os.listdir(src_class_dir)
        random.shuffle(all_images)
        
        # Balance: Take 'images_per_class' OR all images if fewer exist
        files_to_copy = all_images[:images_per_class] if len(all_images) >= images_per_class else all_images
        
        # Progress print for large datasets
        print(f"Processing {class_name}: Copying {len(files_to_copy)} images...")
            
        for img_file in files_to_copy:
            src_file = os.path.join(src_class_dir, img_file)
            dst_file = os.path.join(dst_class_dir, img_file)
            shutil.copy(src_file, dst_file)
            
    print("✅ Balanced whole dataset created.")
else:
    print("✅ Balanced dataset already exists.")

# ===========================================
# STEP 3: DATA GENERATORS
# ===========================================
print("\nInitializing Data Generators...")

if MODEL_CHOICE == 'mobilenet':
    print("Using MobileNetV2 preprocessing (scales -1 to 1).")
    datagen = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess, 
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=val_split
    )
elif MODEL_CHOICE == 'resnet':
    print("Using ResNetV2 preprocessing (scales 0 to 1).")
    datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=val_split
    )
else:
    print(f"ERROR: Unknown MODEL_CHOICE: {MODEL_CHOICE}")
    sys.exit()

# MODIFIED: Removed 'classes=selected_classes' so it scans all subfolders
train_gen = datagen.flow_from_directory(
    balanced_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', # Detects classes automatically
    subset='training', shuffle=True
)

val_gen = datagen.flow_from_directory(
    balanced_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical',
    subset='validation', shuffle=False
)

# Double check class count matches generator detection
if train_gen.num_classes != num_classes:
    print(f"Warning: Generator found {train_gen.num_classes} classes, but we detected {num_classes} folders.")
    num_classes = train_gen.num_classes

# ===========================================
# STEP 4: BUILD MODEL
# ===========================================
print(f"\nBuilding model with {MODEL_CHOICE} base...")
inputs = layers.Input(shape=img_size + (3,))

if MODEL_CHOICE == 'mobilenet':
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
    unfreeze_last_n_layers = 50 
elif MODEL_CHOICE == 'resnet':
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
    unfreeze_last_n_layers = 50 

# --- STAGE 1 SETUP: FREEZE THE BASE ---
base_model.trainable = False

x = base_model(inputs) 
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
# MODIFIED: Dynamic class count
outputs = layers.Dense(num_classes, activation='softmax')(x) 
model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("--- Model Summary (Stage 1: Head Training) ---")
model.summary()

# ===========================================
# STEP 5: TRAIN MODEL - STAGE 1
# ===========================================
print(f"\n--- Starting Model Training: STAGE 1 (Training Head) for {stage_1_epochs} epochs ---")
history_1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=stage_1_epochs
)
print("--- Stage 1 Training Finished ---")

# ===========================================
# STEP 6: TRAIN MODEL - STAGE 2 (Fine-Tuning)
# ===========================================
print("\n--- Re-compiling for STAGE 2 (Fine-Tuning) ---")

base_model.trainable = True
freeze_until_layer = len(base_model.layers) - unfreeze_last_n_layers

print(f"Total layers in base_model: {len(base_model.layers)}")
print(f"Freezing all layers up to layer {freeze_until_layer}...")

for layer in base_model.layers[:freeze_until_layer]:
    layer.trainable = False
    
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n--- Starting Model Training: STAGE 2 (Fine-Tuning) for {stage_2_epochs} epochs ---")
history_2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=total_epochs,
    initial_epoch=history_1.epoch[-1]
)
print("--- Stage 2 Training Finished ---")

# ===========================================
# STEP 7: EVALUATE AND SAVE
# ===========================================
loss, accuracy = model.evaluate(val_gen)
print(f"\n✅ Final Validation Accuracy for {MODEL_CHOICE}: {accuracy * 100:.2f}%")

# MODIFIED: Dynamic filename for the whole dataset
save_path = f"plantvillage_whole_{MODEL_CHOICE}_2stage_weights.h5"
model.save_weights(save_path)
print(f"💾 Model saved successfully to {save_path}!")

# ===========================================
# STEP 8: PLOT
# ===========================================
history = {
    'accuracy': history_1.history['accuracy'] + history_2.history['accuracy'],
    'val_accuracy': history_1.history['val_accuracy'] + history_2.history['val_accuracy'],
    'loss': history_1.history['loss'] + history_2.history['loss'],
    'val_loss': history_1.history['val_loss'] + history_2.history['val_loss'],
}

plt.figure(figsize=(12, 5))
plt.suptitle(f'Whole Dataset Training: {MODEL_CHOICE.upper()}', fontsize=16)

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.axvline(stage_1_epochs - 1, color='red', linestyle='--', label='Start Fine-Tuning')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.axvline(stage_1_epochs - 1, color='red', linestyle='--', label='Start Fine-Tuning')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()