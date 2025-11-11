import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import sys
import shutil
import random

# ===========================================
# USER SETTINGS
# ===========================================
# Original dataset path
original_dataset_dir = r"C:\Users\bosem\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\plantvillage"

# Path for the new balanced dataset
balanced_dir = "tomato_balanced_dataset" 

selected_classes = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus"
]

img_size = (224, 224)
batch_size = 32
val_split = 0.2
images_per_class = 950 # New setting for balancing
epochs = 20 # Increased for fine-tuning

# Set random seed for reproducibility
random.seed(42)

# ===========================================
# STEP 1: VERIFY ORIGINAL DATA
# ===========================================
print("Verifying original dataset path...")
all_files = os.listdir(original_dataset_dir)
available_classes = [d for d in all_files if d in selected_classes]

if len(available_classes) != len(selected_classes):
    print(f"❌ ERROR: Missing classes in {original_dataset_dir}")
    print(f"Found: {available_classes}")
    print(f"Expected: {selected_classes}")
    sys.exit()
else:
    print(f"✅ Found all {len(available_classes)} original class folders.")

# ===========================================
# STEP 2: CREATE BALANCED DATASET
# ===========================================
print(f"\nCreating balanced dataset at: {balanced_dir}")

# Clear the balanced directory if it already exists
if os.path.exists(balanced_dir):
    print("Removing old balanced dataset directory...")
    shutil.rmtree(balanced_dir)

# Create the root directory
os.makedirs(balanced_dir, exist_ok=True)

for class_name in selected_classes:
    print(f"  Processing: {class_name}")
    
    # 1. Create source and destination paths
    src_class_dir = os.path.join(original_dataset_dir, class_name)
    dst_class_dir = os.path.join(balanced_dir, class_name)
    os.makedirs(dst_class_dir, exist_ok=True)
    
    # 2. Get all images and shuffle them
    all_images = os.listdir(src_class_dir)
    random.shuffle(all_images)
    
    # 3. Check if we have enough images
    if len(all_images) < images_per_class:
        print(f"  ⚠️ WARNING: Class '{class_name}' only has {len(all_images)} images. Using all of them.")
        files_to_copy = all_images
    else:
        files_to_copy = all_images[:images_per_class]
        
    # 4. Copy the selected files
    for img_file in files_to_copy:
        src_path = os.path.join(src_class_dir, img_file)
        dst_path = os.path.join(dst_class_dir, img_file)
        shutil.copy(src_path, dst_path)

print(f"✅ Balanced dataset created with {images_per_class} images per class.")

# ===========================================
# STEP 3: DATA GENERATORS
# ===========================================
print("\nInitializing Data Generators from balanced dataset...")
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=val_split
)

# Point the generator at the new BALANCED directory
train_gen = datagen.flow_from_directory(
    balanced_dir,
    target_size=img_size,
    batch_size=batch_size,
    classes=selected_classes, # We know the classes now
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    balanced_dir,
    target_size=img_size,
    batch_size=batch_size,
    classes=selected_classes,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ===========================================
# STEP 4: PRETRAINED MODEL (Fine-Tuning)
# ===========================================
print("\nBuilding fine-tuning model...")
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=img_size + (3,))

# 1. Unfreeze the base model
base_model.trainable = True

# 2. Freeze all layers *except* the last 50
unfreeze_last_n_layers = 50 
freeze_until_layer = len(base_model.layers) - unfreeze_last_n_layers

print(f"Total layers in base_model: {len(base_model.layers)}")
print(f"Freezing all layers up to layer {freeze_until_layer}...")

for layer in base_model.layers[:freeze_until_layer]:
    layer.trainable = False

# 3. Build model with Functional API (to freeze BN layers)
inputs = layers.Input(shape=img_size + (3,))
# Call base_model with training=False to keep Batch Norm layers in inference mode
x = base_model(inputs, training=False) 
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(selected_classes), activation='softmax')(x)
model = Model(inputs, outputs)

# 4. Compile with a low learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-4), # 0.0001
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===========================================
# STEP 5: TRAIN MODEL
# ===========================================
print("\n--- Starting Model Training ---")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)
print("--- Model Training Finished ---")

# ===========================================
# STEP 6: EVALUATE AND SAVE
# ===========================================
loss, accuracy = model.evaluate(val_gen)
print(f"\n✅ Final Validation Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Final Validation Loss: {loss:.4f}")

model.save_weights("tomato_disease_finetune_balanced_weights.h5")
print("💾 Model saved successfully!")

# ===========================================
# STEP 7: PLOT ACCURACY & LOSS CURVES
# ===========================================
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
