import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications import MobileNetV2, ResNet50V2
from tensorflow.keras import layers
import numpy as np
import os
import shutil
import scipy.stats

# ===========================================
# CONFIGURATION
# ===========================================
# Must match your trained model
MODEL_CHOICE = 'resnet' 
WEIGHTS_FILE = "E:\Study Materials\Projects\Curriculum Learning\MultiClass\TeacherModel\plantvillage_whole_resnet_2stage_weights.h5"
SOURCE_DATA_DIR = "E:\Study Materials\Projects\Curriculum Learning\MultiClass\TeacherModel\plantvillage_whole_balanced"
TARGET_DATA_DIR = "E:\Study Materials\Projects\Curriculum Learning\MultiClass\StudentModel\plantvillage_curriculum"

# Curriculum Splits (33% Easy, 33% Medium, 34% Hard)
SPLIT_RATIOS = [0.33, 0.33, 0.34] 

IMG_SIZE = (224, 224)
BATCH_SIZE = 64 # Larger batch for faster inference

# ===========================================
# 1. LOAD TRAINED TEACHER MODEL
# ===========================================
print(f"--- Loading Teacher Model: {MODEL_CHOICE.upper()} ---")

# Dynamically find classes
classes = sorted([d for d in os.listdir(SOURCE_DATA_DIR) 
                  if os.path.isdir(os.path.join(SOURCE_DATA_DIR, d))])
num_classes = len(classes)
print(f"Detected {num_classes} classes.")

# Rebuild Architecture
inputs = layers.Input(shape=IMG_SIZE + (3,))
if MODEL_CHOICE == 'mobilenet':
    base = MobileNetV2(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))
    preprocess_func = mobilenet_preprocess
elif MODEL_CHOICE == 'resnet':
    base = ResNet50V2(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))
    # ResNet uses 0-1 scaling, handled manually in loop if needed
    preprocess_func = lambda x: x / 255.0

# Reconstruct the Head
x = base(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

# Load Weights
model.load_weights(WEIGHTS_FILE)
print("✅ Weights loaded.")

# ===========================================
# 2. SCORING FUNCTION
# ===========================================
def calculate_entropy(predictions):
    """
    Computes Shannon Entropy for a batch of predictions.
    Higher Entropy = Higher Uncertainty = Harder Sample.
    """
    return scipy.stats.entropy(predictions, axis=1)

# ===========================================
# 3. PROCESSING LOOP
# ===========================================
print(f"\n--- Starting Difficulty Estimation ---")

# Prepare directories: D1 (Easy), D2 (Medium), D3 (Hard)
stages = ['D1_Easy', 'D2_Medium', 'D3_Hard']
for stage in stages:
    stage_path = os.path.join(TARGET_DATA_DIR, stage)
    if os.path.exists(stage_path):
        shutil.rmtree(stage_path) # Clean start
    os.makedirs(stage_path)
    # Create class subfolders in each stage
    for cls in classes:
        os.makedirs(os.path.join(stage_path, cls))

# Iterate per class to ensure class balance in every stage
for class_name in classes:
    print(f"Processing Class: {class_name}...")
    class_dir = os.path.join(SOURCE_DATA_DIR, class_name)
    image_files = os.listdir(class_dir)
    
    # Load and Preprocess batch
    # (Loading one by one to avoid OOM, but grouped in lists)
    filepaths = []
    batch_images = []
    
    for fname in image_files:
        fpath = os.path.join(class_dir, fname)
        img = load_img(fpath, target_size=IMG_SIZE)
        img_arr = img_to_array(img)
        
        # Apply correct preprocessing
        if MODEL_CHOICE == 'resnet':
            img_arr = img_arr / 255.0
        elif MODEL_CHOICE == 'mobilenet':
            img_arr = mobilenet_preprocess(img_arr)
            
        batch_images.append(img_arr)
        filepaths.append(fname)
        
    # Predict
    batch_images = np.array(batch_images)
    preds = model.predict(batch_images, batch_size=BATCH_SIZE, verbose=0)
    
    # Calculate Entropy (Difficulty Score)
    entropies = calculate_entropy(preds)
    
    # Pair: (Filename, Entropy)
    scored_files = list(zip(filepaths, entropies))
    
    # Sort by Entropy: Low (Easy) -> High (Hard)
    scored_files.sort(key=lambda x: x[1])
    
    # Split into buckets
    n = len(scored_files)
    idx1 = int(n * SPLIT_RATIOS[0])
    idx2 = int(n * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))
    
    easy_batch = scored_files[:idx1]
    medium_batch = scored_files[idx1:idx2]
    hard_batch = scored_files[idx2:]
    
    # Helper to copy
    def copy_files(file_list, stage_name):
        dst_dir = os.path.join(TARGET_DATA_DIR, stage_name, class_name)
        for fname, score in file_list:
            src = os.path.join(class_dir, fname)
            dst = os.path.join(dst_dir, fname)
            shutil.copy(src, dst)
            
    copy_files(easy_batch, 'D1_Easy')
    copy_files(medium_batch, 'D2_Medium')
    copy_files(hard_batch, 'D3_Hard')

print("\n✅ Curriculum Dataset Generation Complete!")
print(f"Location: {os.path.abspath(TARGET_DATA_DIR)}")
print("Structure:")
print(f"  ├── D1_Easy   (Lowest Entropy)")
print(f"  ├── D2_Medium (Mid Entropy)")
print(f"  └── D3_Hard   (Highest Entropy)")