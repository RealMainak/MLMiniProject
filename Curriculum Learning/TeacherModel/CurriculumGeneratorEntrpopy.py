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
import matplotlib.pyplot as plt
import json

# ===========================================
# CONFIGURATION (STRICT PATHS)
# ===========================================
MODEL_CHOICE = 'resnet' 
WEIGHTS_FILE = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\TeacherModel\plantvillage_whole_resnet_2stage_weights.h5"
SOURCE_DATA_DIR = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\TeacherModel\plantvillage_whole_balanced"
TARGET_DATA_DIR = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\StudentModel\plantvillage_curriculum_standard"

SPLIT_RATIOS = [0.33, 0.33, 0.34] 
IMG_SIZE = (224, 224)

# REDUCED BATCH SIZE FOR MX130 GPU
BATCH_SIZE = 8 
NUM_CLASSES = 15 

# ===========================================
# 1. BUILD MODEL
# ===========================================
print(f"--- Building Model for {NUM_CLASSES} classes ---")

inputs = layers.Input(shape=IMG_SIZE + (3,), name='input_1')
base = ResNet50V2(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))
base._name = 'resnet50v2'
x = base(inputs)
x = layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
x = layers.Dropout(0.4, name='dropout')(x)
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
print(f"\n--- Starting Difficulty Estimation ---")

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

# Store entropy scores for visualization
all_entropy_scores = {}

def calculate_entropy(predictions):
    return scipy.stats.entropy(predictions, axis=1)

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
            
        # 3. Predict immediately
        batch_arr = np.array(batch_images)
        preds = model.predict(batch_arr, verbose=0)
        
        # 4. Calculate Entropy
        entropies = calculate_entropy(preds)
        
        # 5. Store results and discard images
        for fname, score in zip(valid_filenames, entropies):
            class_scores.append((fname, score))
            
        # Optional: Explicitly free memory
        del batch_images
        del batch_arr
        
        # Progress indicator
        print(f"   Processed {min(i + BATCH_SIZE, total_files)}/{total_files} images", end='\r')
    
    print("") # Newline after progress bar
    
    # Store entropy scores for all classes
    all_entropy_scores[class_name] = [score for _, score in class_scores]
    
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

# ===========================================
# 4. VISUALIZE ENTROPY DISTRIBUTIONS
# ===========================================
print("\n--- Generating Entropy Distribution Visualizations ---")

# Save entropy scores to JSON for future reference
entropy_stats = {}
for class_name, scores in all_entropy_scores.items():
    entropy_stats[class_name] = {
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'mean': float(np.mean(scores)),
        'median': float(np.median(scores)),
        'std': float(np.std(scores)),
        'count': len(scores)
    }

stats_file = os.path.join(TARGET_DATA_DIR, '../entropy_statistics.json')
with open(stats_file, 'w') as f:
    json.dump(entropy_stats, f, indent=2)
print(f"✅ Entropy statistics saved to {stats_file}")

# Create visualization: Histograms
fig, axes = plt.subplots(5, 3, figsize=(16, 20))
axes = axes.flatten()

threshold1 = SPLIT_RATIOS[0]
threshold2 = SPLIT_RATIOS[0] + SPLIT_RATIOS[1]

for idx, class_name in enumerate(sorted(all_entropy_scores.keys())):
    if idx >= 15:
        break
    
    scores = all_entropy_scores[class_name]
    sorted_scores = sorted(scores)
    
    # Calculate threshold values
    n = len(sorted_scores)
    thresh1_idx = int(n * threshold1)
    thresh2_idx = int(n * threshold2)
    thresh1_val = sorted_scores[thresh1_idx]
    thresh2_val = sorted_scores[thresh2_idx]
    
    ax = axes[idx]
    ax.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(thresh1_val, color='green', linestyle='--', linewidth=2, label=f'Easy/Medium ({thresh1_val:.3f})')
    ax.axvline(thresh2_val, color='orange', linestyle='--', linewidth=2, label=f'Medium/Hard ({thresh2_val:.3f})')
    ax.set_xscale('log')
    ax.set_title(f'{class_name}\n(n={len(scores)})', fontsize=10, fontweight='bold')
    ax.set_xlabel('Entropy Score (log scale)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

# Hide extra subplots
for idx in range(len(all_entropy_scores), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
hist_file = os.path.join(TARGET_DATA_DIR, '../entropy_histograms.png')
plt.savefig(hist_file, dpi=150, bbox_inches='tight')
print(f"✅ Histograms saved to {hist_file}")
plt.close()

# Create visualization: Box plots for all classes
fig, ax = plt.subplots(figsize=(14, 6))
box_data = [all_entropy_scores[cls] for cls in sorted(all_entropy_scores.keys())]
bp = ax.boxplot(box_data, labels=sorted(all_entropy_scores.keys()), patch_artist=True)

for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    
ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Entropy Score (log scale)', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.set_title('Entropy Score Distribution by Class (Exponential Scale)', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.3, axis='y', which='both')
plt.tight_layout()

boxplot_file = os.path.join(TARGET_DATA_DIR, '../entropy_boxplots.png')
plt.savefig(boxplot_file, dpi=150, bbox_inches='tight')
print(f"✅ Box plots saved to {boxplot_file}")
plt.close()

# Create visualization: Overall distribution with thresholds
fig, ax = plt.subplots(figsize=(12, 6))
all_scores = np.concatenate(list(all_entropy_scores.values()))
sorted_all = np.sort(all_scores)
thresh1_val = sorted_all[int(len(sorted_all) * threshold1)]
thresh2_val = sorted_all[int(len(sorted_all) * threshold2)]

ax.hist(all_scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(thresh1_val, color='green', linestyle='--', linewidth=3, label=f'Easy/Medium Threshold ({thresh1_val:.3f})')
ax.axvline(thresh2_val, color='red', linestyle='--', linewidth=3, label=f'Medium/Hard Threshold ({thresh2_val:.3f})')
ax.fill_between([all_scores.min(), thresh1_val], 0, ax.get_ylim()[1], alpha=0.1, color='green', label='Easy Region')
ax.fill_between([thresh1_val, thresh2_val], 0, ax.get_ylim()[1], alpha=0.1, color='yellow', label='Medium Region')
ax.fill_between([thresh2_val, all_scores.max()], 0, ax.get_ylim()[1], alpha=0.1, color='red', label='Hard Region')

ax.set_xscale('log')
ax.set_xlabel('Entropy Score (log scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title(f'Overall Entropy Distribution - Exponential Scale (Total: {len(all_scores)} images)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, which='both')
plt.tight_layout()

overall_file = os.path.join(TARGET_DATA_DIR, '../entropy_overall_distribution.png')
plt.savefig(overall_file, dpi=150, bbox_inches='tight')
print(f"✅ Overall distribution saved to {overall_file}")
plt.close()

print("\n📊 All visualizations created successfully!")
print(f"   - Histograms per class: {hist_file}")
print(f"   - Box plots: {boxplot_file}")
print(f"   - Overall distribution: {overall_file}")
print(f"   - Statistics JSON: {stats_file}")