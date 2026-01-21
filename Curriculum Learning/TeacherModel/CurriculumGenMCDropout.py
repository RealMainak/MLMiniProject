import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import layers
import numpy as np
import os
import shutil
import scipy.stats
import sys
import h5py
import gc # Garbage Collection

# ===========================================
# CONFIGURATION (STRICT PATHS)
# ===========================================
MODEL_CHOICE = 'resnet' 
WEIGHTS_FILE = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\TeacherModel\plantvillage_whole_resnet_2stage_weights.h5"
SOURCE_DATA_DIR = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\TeacherModel\plantvillage_whole_balanced"
TARGET_DATA_DIR = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\StudentModel\plantvillage_curriculum_mcdropout"

SPLIT_RATIOS = [0.33, 0.33, 0.34] 
IMG_SIZE = (224, 224)

# REDUCED BATCH SIZE FOR MX130 GPU
BATCH_SIZE = 8 
NUM_CLASSES = 15 

# MC DROPOUT SETTINGS
MC_ITERATIONS = 20  # Number of forward passes per image. Higher = more accurate uncertainty, but slower.

# ===========================================
# 0. CUSTOM LAYER DEFINITION
# ===========================================
class MCDropout(layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

# ===========================================
# 1. BUILD MODEL
# ===========================================
print(f"--- Building Model for {NUM_CLASSES} classes with MC Dropout ---")

inputs = layers.Input(shape=IMG_SIZE + (3,), name='input_1')
base = ResNet50V2(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))
base._name = 'resnet50v2'
x = base(inputs)
x = layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)

# REPLACED STANDARD DROPOUT WITH MC DROPOUT
x = MCDropout(0.4, name='dropout')(x) 

outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='dense')(x)

model = Model(inputs, outputs)
print("✅ Model built successfully.")

# ===========================================
# 2. MANUAL WEIGHT LOADING
# ===========================================
print(f"--- Loading weights manually ---")

def manual_load(model, filepath):
    with h5py.File(filepath, 'r') as f:
        # Load Dense
        try:
            if 'dense' in f and 'dense' in f['dense']:
                dense_g = f['dense']['dense']
                kernel = dense_g['kernel:0'][:]
                bias = dense_g['bias:0'][:]
                model.get_layer('dense').set_weights([kernel, bias])
        except Exception as e:
            print(f"   ⚠️ Warning: Dense layer issue: {e}")

        # Load ResNet
        try:
            base_model = model.get_layer('resnet50v2')
            if 'resnet50v2' in f:
                resnet_group = f['resnet50v2']
                for layer in base_model.layers:
                    if layer.name in resnet_group:
                        try:
                            g = resnet_group[layer.name]
                            if layer.name in g: g = g[layer.name]
                            layer_weights = []
                            for w_tensor in layer.weights:
                                w_name = w_tensor.name.split('/')[-1]
                                if w_name in g:
                                    layer_weights.append(g[w_name][:])
                            if layer_weights:
                                layer.set_weights(layer_weights)
                        except Exception: pass
        except ValueError: pass

manual_load(model, WEIGHTS_FILE)
print("✅ Weights loaded.")

# ===========================================
# 3. MEMORY-OPTIMIZED PROCESSING LOOP
# ===========================================
print(f"\n--- Starting MC Dropout Difficulty Estimation (T={MC_ITERATIONS}) ---")

folder_classes = sorted([d for d in os.listdir(SOURCE_DATA_DIR) 
                  if os.path.isdir(os.path.join(SOURCE_DATA_DIR, d))])

# Prepare directories
stages = ['D1_Easy', 'D2_Medium', 'D3_Hard']
for stage in stages:
    stage_path = os.path.join(TARGET_DATA_DIR, stage)
    if os.path.exists(stage_path):
        shutil.rmtree(stage_path) 
    os.makedirs(stage_path)
    for cls in folder_classes:
        os.makedirs(os.path.join(stage_path, cls))

def calculate_entropy(predictions):
    """
    Calculates predictive entropy based on the MEAN prediction across MC samples.
    """
    # predictions shape: (batch_size, num_classes)
    # Add epsilon to prevent log(0)
    epsilon = 1e-10
    return -np.sum(predictions * np.log(predictions + epsilon), axis=1)

# Iterate Classes
for class_name in folder_classes:
    print(f"Processing Class: {class_name}...")
    class_dir = os.path.join(SOURCE_DATA_DIR, class_name)
    image_files = os.listdir(class_dir)
    
    # Store tuples of (filename, entropy_score)
    class_scores = []
    
    # PROCESS IN CHUNKS to save RAM/GPU Memory
    total_files = len(image_files)
    for i in range(0, total_files, BATCH_SIZE):
        # 1. Select a small batch of files
        batch_files = image_files[i : i + BATCH_SIZE]
        batch_images = []
        valid_filenames = []
        
        # 2. Load images into memory
        for fname in batch_files:
            fpath = os.path.join(class_dir, fname)
            try:
                img = load_img(fpath, target_size=IMG_SIZE)
                img_arr = img_to_array(img)
                if MODEL_CHOICE == 'resnet':
                    img_arr = img_arr / 255.0 
                batch_images.append(img_arr)
                valid_filenames.append(fname)
            except Exception:
                continue
        
        if not batch_images:
            continue
            
        # 3. MC DROPOUT PREDICTION
        batch_arr = np.array(batch_images)
        
        # We need to collect predictions over multiple iterations
        # Shape: (MC_ITERATIONS, batch_size, num_classes)
        mc_predictions = []
        
        for _ in range(MC_ITERATIONS):
            # model(x, training=True) is implicitly handled by our MCDropout layer
            # but standard predict() works too because we overrode call()
            pred = model.predict(batch_arr, verbose=0)
            mc_predictions.append(pred)
            
        # Stack to (MC_ITERATIONS, batch_size, num_classes)
        mc_predictions = np.stack(mc_predictions)
        
        # 4. Average the predictions (Mean Probability)
        # Shape: (batch_size, num_classes)
        mean_prediction = np.mean(mc_predictions, axis=0)
        
        # 5. Calculate Entropy on the Mean Prediction
        entropies = calculate_entropy(mean_prediction)
        
        # 6. Store results and discard images
        for fname, score in zip(valid_filenames, entropies):
            class_scores.append((fname, score))
            
        # Explicitly free memory
        del batch_images
        del batch_arr
        del mc_predictions
        del mean_prediction
        
        # Progress indicator
        print(f"   Processed {min(i + BATCH_SIZE, total_files)}/{total_files} images", end='\r')
    
    print("") # Newline after progress bar
    
    # Sort by Entropy (Easy -> Hard)
    class_scores.sort(key=lambda x: x[1])
    
    # Split
    n = len(class_scores)
    idx1 = int(n * SPLIT_RATIOS[0])
    idx2 = int(n * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))
    
    easy_batch = class_scores[:idx1]
    medium_batch = class_scores[idx1:idx2]
    hard_batch = class_scores[idx2:]
    
    # Copy files (Using copy2 to preserve metadata)
    def copy_files(file_list, stage_name):
        dst_dir = os.path.join(TARGET_DATA_DIR, stage_name, class_name)
        for fname, score in file_list:
            src = os.path.join(class_dir, fname)
            dst = os.path.join(dst_dir, fname)
            shutil.copy2(src, dst)
            
    copy_files(easy_batch, 'D1_Easy')
    copy_files(medium_batch, 'D2_Medium')
    copy_files(hard_batch, 'D3_Hard')
    
    # Force garbage collection between classes
    gc.collect()

print("\n✅ Curriculum Dataset Generation Complete!")
print(f"Location: {TARGET_DATA_DIR}")