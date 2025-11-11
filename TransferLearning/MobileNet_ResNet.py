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
# MODIFIED: Removed 'from google.colab import files'

# ===========================================
# STEP 0: LOCAL SETUP
# ===========================================
print("TensorFlow Version:", tf.__version__)
print("Available GPU(s):", tf.config.list_physical_devices('GPU'))

# MODIFIED: The Colab !pip, !kaggle, and file upload commands 
# have been removed. Please follow the manual setup instructions above.


# ===========================================
# USER SETTINGS & PATHS
# ===========================================
# --- !!! CHOOSE YOUR MODEL HERE !!! ---
# Run once with 'mobilenet', then change to 'resnet' and run again.
MODEL_CHOICE = 'mobilenet'  # Options: 'mobilenet', 'resnet'
# ------------------------------------

# --- PATHS UPDATED FOR LOCAL PC ---
# MODIFIED: 
# !!! YOU MUST CHANGE THIS PATH !!!
# Set this to the full path of the 'PlantVillage' folder you unzipped.
# Example (Windows): "C:/Users/YourName/Downloads/PlantVillage"
# Example (Linux/Mac): "/home/YourName/datasets/PlantVillage"
original_dataset_dir = r"C:\Users\bosem\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\plantvillage" 

# This will create a new folder in the same directory as your script
balanced_dir = "tomato_balanced_dataset" 
# -------------------------------

selected_classes = [
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", 
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus"
]
img_size = (224, 224)
batch_size = 32
val_split = 0.2
images_per_class = 950
stage_1_epochs = 15
stage_2_epochs = 30
total_epochs = stage_1_epochs + stage_2_epochs
random.seed(42)

print(f"--- INITIALIZING EXPERIMENT FOR: {MODEL_CHOICE} ---")

# ===========================================
# STEP 1 & 2: CREATE BALANCED DATASET 
# (This part is unchanged and will be skipped if dir exists)
# ===========================================

# Quick check to ensure the user updated the path
if "CHANGE/ME" in original_dataset_dir:
    print("="*50)
    print("ERROR: Please update the 'original_dataset_dir' variable")
    print("       to point to your unzipped 'PlantVillage' folder.")
    print("="*50)
    sys.exit()

print(f"Checking for balanced dataset at: {balanced_dir}")
if not os.path.exists(balanced_dir):
    print(f"Creating balanced dataset at: {balanced_dir}")
    os.makedirs(balanced_dir, exist_ok=True)
    for class_name in selected_classes:
        src_class_dir = os.path.join(original_dataset_dir, class_name)
        
        # Check if original class folder exists
        if not os.path.exists(src_class_dir):
            print(f"Warning: Source folder not found, skipping: {src_class_dir}")
            continue
            
        dst_class_dir = os.path.join(balanced_dir, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)
        
        all_images = os.listdir(src_class_dir)
        random.shuffle(all_images)
        
        files_to_copy = all_images[:images_per_class] if len(all_images) >= images_per_class else all_images
            
        for img_file in files_to_copy:
            shutil.copy(os.path.join(src_class_dir, img_file), 
                        os.path.join(dst_class_dir, img_file))
    print("✅ Balanced dataset created.")
else:
    print("✅ Balanced dataset already exists.")

# ===========================================
# STEP 3: DATA GENERATORS
# (Logic unchanged, paths are now local)
# ===========================================
print("\nInitializing Data Generators...")

# --- CRITICAL: Use the correct pre-processing for each model ---
if MODEL_CHOICE == 'mobilenet':
    print("Using MobileNetV2 preprocessing (scales -1 to 1).")
    datagen = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess, # Scales -1 to 1
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=val_split
    )
elif MODEL_CHOICE == 'resnet':
    print("Using ResNetV2 preprocessing (scales 0 to 1).")
    datagen = ImageDataGenerator(
        rescale=1./255, # Scales 0 to 1
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=val_split
    )
else:
    print(f"ERROR: Unknown MODEL_CHOICE: {MODEL_CHOICE}")
    sys.exit()
# ---------------------------------------------------------------

train_gen = datagen.flow_from_directory(
    balanced_dir, target_size=img_size, batch_size=batch_size,
    classes=selected_classes, class_mode='categorical',
    subset='training', shuffle=True
)

val_gen = datagen.flow_from_directory(
    balanced_dir, target_size=img_size, batch_size=batch_size,
    classes=selected_classes, class_mode='categorical',
    subset='validation', shuffle=False
)

# ===========================================
# STEP 4: BUILD MODEL
# (Logic unchanged)
# ===========================================
print(f"\nBuilding model with {MODEL_CHOICE} base...")
inputs = layers.Input(shape=img_size + (3,))

# --- Load the correct base model ---
if MODEL_CHOICE == 'mobilenet':
    # CORRECT
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
    unfreeze_last_n_layers = 50 # MobileNetV2 has 154 layers
elif MODEL_CHOICE == 'resnet':
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
    unfreeze_last_n_layers = 50 # ResNet50V2 has 190 layers
# -----------------------------------

# --- STAGE 1 SETUP: FREEZE THE BASE ---
base_model.trainable = False

x = base_model(inputs) 
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(selected_classes), activation='softmax')(x)
model = Model(inputs, outputs)

# Compile for Stage 1
model.compile(
    optimizer=Adam(learning_rate=1e-3), # 0.001
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("--- Model Summary (Stage 1: Head Training) ---")
model.summary()

# ===========================================
# STEP 5: TRAIN MODEL - STAGE 1 (Train the Head)
# (Logic unchanged)
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
# (Logic unchanged)
# ===========================================
print("\n--- Re-compiling for STAGE 2 (Fine-Tuning) ---")

base_model.trainable = True
freeze_until_layer = len(base_model.layers) - unfreeze_last_n_layers

print(f"Total layers in base_model: {len(base_model.layers)}")
print(f"Freezing all layers up to layer {freeze_until_layer}...")

for layer in base_model.layers[:freeze_until_layer]:
    layer.trainable = False
    
# CRITICAL: Manually freeze all Batch Norm layers
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Re-compile with a very low learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5), # 0.00001
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("--- Model Summary (Stage 2: Fine-Tuning) ---")
model.summary()

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
print(f"✅ Final Validation Loss for {MODEL_CHOICE}: {loss:.4f}")

# Save with a dynamic file name
save_path = f"tomato_disease_{MODEL_CHOICE}_2stage_weights.h5"
model.save_weights(save_path)
print(f"💾 Model saved successfully to {save_path}!")

# --- MODIFIED: REMOVED COLAB DOWNLOAD ---
# The file is already saved to your local disk in the
# same folder where you ran the Python script.
# -------------------------------------------

# ===========================================
# STEP 8: PLOT COMBINED ACCURACY & LOSS
# (Logic unchanged)
# ===========================================
history = {
    'accuracy': history_1.history['accuracy'] + history_2.history['accuracy'],
    'val_accuracy': history_1.history['val_accuracy'] + history_2.history['val_accuracy'],
    'loss': history_1.history['loss'] + history_2.history['loss'],
    'val_loss': history_1.history['val_loss'] + history_2.history['val_loss'],
}

plt.figure(figsize=(12, 5))
plt.suptitle(f'Training History for {MODEL_CHOICE.upper()}', fontsize=16)

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
# MODIFIED: plt.show() will open the plot in a new window on your PC
plt.show()
