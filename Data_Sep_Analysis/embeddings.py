import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import ResNet50V2
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*50)
print("STARTING STANDALONE DATASET SEPARABILITY ANALYSIS")
print("="*50)

# ===========================================
# 1. CONFIGURATION & PATHS
# ===========================================
# Update these if your paths or parameters change

BALANCED_DIR = "E:\Study Materials\Projects\ML_PlantVillage\Curriculum Learning\TeacherModel\plantvillage_whole_balanced"
WEIGHTS_PATH = "E:\Study Materials\Projects\ML_PlantVillage\Curriculum Learning\TeacherModel\plantvillage_whole_resnet_2stage_weights.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Cannot find weights at {WEIGHTS_PATH}. Please ensure you ran the training script.")

# ===========================================
# 2. RECREATE VALIDATION GENERATOR
# ===========================================
print("Initializing Validation Generator...")
# Using the exact same preprocessing as your training script for ResNet
datagen = ImageDataGenerator(rescale=1./255)

# shuffle=False is strictly required to align extracted features with true labels
val_gen = datagen.flow_from_directory(
    BALANCED_DIR, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset=None, # We just point it to the directory, but if you split it earlier, ensure it points to validation data
    shuffle=False 
)

num_classes = val_gen.num_classes
class_indices = val_gen.class_indices
class_names = {v: k for k, v in class_indices.items()}

# ===========================================
# 3. REBUILD MODEL SKELETON & LOAD WEIGHTS
# ===========================================
print(f"Detected {num_classes} classes from generator.")
print("Rebuilding ResNet50V2 architecture...")

inputs = layers.Input(shape=IMG_SIZE + (3,))

# 1. Use weights='imagenet' to ensure the internal graph builds identically to training
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))

# 2. MATCH THE TRAINING SCRIPT'S FREEZE STATE EXACTLY
# This ensures the internal weight lists align perfectly with the saved .h5 file
base_model.trainable = True
unfreeze_last_n_layers = 50 
freeze_until_layer = len(base_model.layers) - unfreeze_last_n_layers

for layer in base_model.layers[:freeze_until_layer]:
    layer.trainable = False
    
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# 3. Build the classification head
x = base_model(inputs) 
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x) 
model = Model(inputs, outputs)

print("Loading trained weights...")
# 4. Load weights (by_name=True adds an extra layer of safety against mismatches)
model.load_weights(WEIGHTS_PATH, by_name=True)
print("Weights loaded successfully!")

# ===========================================
# 4. CREATE FEATURE EXTRACTOR
# ===========================================
# Target the GlobalAveragePooling2D layer
gap_layer_output = model.layers[-3].output
feature_extractor = Model(inputs=model.input, outputs=gap_layer_output)
print(f"Feature extractor built targeting layer: {model.layers[-3].name}")

# ===========================================
# 5. EXTRACT FEATURES & APPLY PCA
# ===========================================
print("Extracting embeddings from the dataset (this may take a moment)...")
raw_embeddings = feature_extractor.predict(val_gen)
true_labels = val_gen.classes

print(f"Extracted raw feature matrix shape: {raw_embeddings.shape}")

print("Applying PCA to retain 95% variance for covariance stability...")
pca = PCA(n_components=0.95)
reduced_embeddings = pca.fit_transform(raw_embeddings)

print(f"PCA-reduced feature matrix shape: {reduced_embeddings.shape}")

# ===========================================
# 6. COMPUTE INTRA-CLASS VARIANCE
# ===========================================
# Lower variance indicates tighter clustering and better class consistency[cite: 12].
print("\n--- 3.1 Intra-Class Variance ---")
intra_class_variances = {}

for class_idx in range(num_classes):
    class_mask = (true_labels == class_idx)
    class_features = reduced_embeddings[class_mask]
    
    if len(class_features) > 0:
        var = np.var(class_features, axis=0).mean()
        intra_class_variances[class_names[class_idx]] = var

variance_df = pd.DataFrame(list(intra_class_variances.items()), columns=['Class', 'Variance'])
variance_df = variance_df.sort_values(by='Variance', ascending=True)

print("Top 5 Classes with LOWEST Intra-Class Variance (Tighter clustering):")
print(variance_df.head(5).to_string(index=False))
print("\nTop 5 Classes with HIGHEST Intra-Class Variance (Looser clustering):")
print(variance_df.tail(5).to_string(index=False))

# ===========================================
# 7. COMPUTE INTER-CLASS CENTROID DISTANCES
# ===========================================
# Small centroid distances indicate potential confusion between classes[cite: 14].
print("\n--- 3.2 Inter-Class Centroid Distances ---")
class_centroids = []
ordered_class_names = []

for class_idx in range(num_classes):
    class_mask = (true_labels == class_idx)
    class_features = reduced_embeddings[class_mask]
    if len(class_features) > 0:
        centroid = np.mean(class_features, axis=0)
        class_centroids.append(centroid)
        ordered_class_names.append(class_names[class_idx])

class_centroids = np.array(class_centroids)
centroid_distances = squareform(pdist(class_centroids, metric='euclidean'))

np.fill_diagonal(centroid_distances, np.inf)

print("Top 10 Most Confusable Class Pairs (Smallest centroid distance):")
flat_indices = np.argsort(centroid_distances, axis=None)

pairs_found = 0
seen_pairs = set()

for idx in flat_indices:
    if pairs_found >= 10:
        break
    i, j = np.unravel_index(idx, centroid_distances.shape)
    pair = tuple(sorted((i, j)))
    
    if pair not in seen_pairs:
        seen_pairs.add(pair)
        dist = centroid_distances[i, j]
        print(f"Distance {dist:.4f} : {ordered_class_names[i]} <--> {ordered_class_names[j]}")
        pairs_found += 1

# Create a copy for plotting so we don't mess up the original matrix
plot_matrix = centroid_distances.copy()

# Replace infinity with NaN so Seaborn can calculate the color scale correctly
plot_matrix[np.isinf(plot_matrix)] = np.nan

# A (12, 10) figure is perfect for 15 classes
plt.figure(figsize=(12, 10))

# Pass the ordered_class_names to the tick labels
sns.heatmap(
    plot_matrix, 
    xticklabels=ordered_class_names, 
    yticklabels=ordered_class_names, 
    cmap='viridis_r',
    cbar_kws={'label': 'Euclidean Distance'},
    annot=True, # Optional: Adds the exact distance numbers inside the squares
    fmt=".1f"   # Formats those numbers to 1 decimal place
)

plt.title('Inter-Class Centroid Distance Heatmap\n(Darker = Closer/More Confusable)', fontsize=16, pad=20)

# Rotate the x-axis labels 90 degrees so they don't overlap
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("SEPARABILITY ANALYSIS (PART 1) COMPLETE")
print("="*50)