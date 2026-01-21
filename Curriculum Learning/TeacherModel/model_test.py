import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import MobileNetV2, ResNet50V2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ===========================================
# USER SETTINGS
# ===========================================
# --- CHOOSE THE MODEL TO ANALYZE ---
MODEL_CHOICE = 'resnet'  # Options: 'mobilenet', 'resnet'
# -----------------------------------

balanced_dir = "plantvillage_whole_balanced"
img_size = (224, 224)
batch_size = 32
val_split = 0.2

# ===========================================
# SETUP & CHECKS
# ===========================================
print(f"--- INITIALIZING ANALYSIS FOR: {MODEL_CHOICE.upper()} ---")

if not os.path.exists(balanced_dir):
    print(f"ERROR: Directory '{balanced_dir}' not found.")
    sys.exit()

# Dynamically detect classes
class_names = sorted([d for d in os.listdir(balanced_dir) 
                      if os.path.isdir(os.path.join(balanced_dir, d))])
num_classes = len(class_names)
print(f"Detected {num_classes} classes.")

# ===========================================
# MODEL BUILDING FUNCTION
# ===========================================
def build_model(model_name, num_classes):
    inputs = layers.Input(shape=img_size + (3,)) 
    
    if model_name == 'mobilenet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
        unfreeze_last_n_layers = 50
    elif model_name == 'resnet':
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
        unfreeze_last_n_layers = 50
    else:
        raise ValueError("Unknown model_name")
        
    # Replicate exact architecture
    base_model.trainable = True
    freeze_until_layer = len(base_model.layers) - unfreeze_last_n_layers

    for layer in base_model.layers[:freeze_until_layer]:
        layer.trainable = False
    
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            
    x = base_model(inputs, training=False) 
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ===========================================
# PREPARATION & LOADING
# ===========================================
# 1. Configure Data Generator based on Model Choice
if MODEL_CHOICE == 'mobilenet':
    print("Using MobileNetV2 preprocessing.")
    datagen = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess,
        validation_split=val_split
    )
    weights_filename = "plantvillage_whole_mobilenet_2stage_weights.h5"

elif MODEL_CHOICE == 'resnet':
    print("Using ResNet50V2 preprocessing.")
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split
    )
    weights_filename = "plantvillage_whole_resnet_2stage_weights.h5"
else:
    print(f"Error: Unknown MODEL_CHOICE '{MODEL_CHOICE}'")
    sys.exit()

# 2. Create Validation Generator
val_gen = datagen.flow_from_directory(
    balanced_dir, target_size=img_size, batch_size=batch_size,
    class_mode='categorical',
    subset='validation', shuffle=False
)

# 3. Build & Load Weights
print(f"\nLoading {weights_filename}...")
if not os.path.exists(weights_filename):
    print(f"CRITICAL ERROR: Weight file '{weights_filename}' not found.")
    print("Please run the training script first.")
    sys.exit()

model = build_model(MODEL_CHOICE, num_classes)
model.load_weights(weights_filename)
print("✅ Model loaded successfully.")

# ===========================================
# INFERENCE & ANALYSIS
# ===========================================
print("\n--- Running Inference ---")
# Get predictions
val_gen.reset()
predictions = model.predict(val_gen, steps=len(val_gen), verbose=1)
predicted_indices = np.argmax(predictions, axis=1)
true_indices = val_gen.classes
labels = list(val_gen.class_indices.keys())

# 1. Overall Metrics
loss, acc = model.evaluate(val_gen, verbose=0)
print(f"\n📊 Global Accuracy: {acc*100:.2f}%")
print(f"📊 Global Loss:     {loss:.4f}")

# 2. Detailed Classification Report
print("\n--- Detailed Classification Report ---")
# This shows Precision, Recall, and F1-Score for EVERY class
report = classification_report(true_indices, predicted_indices, target_names=labels)
print(report)

# 3. Confusion Matrix Plot
print("\n--- Generating Confusion Matrix ---")
cm = confusion_matrix(true_indices, predicted_indices)

plt.figure(figsize=(24, 20))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 8})

plt.title(f'Confusion Matrix: {MODEL_CHOICE.upper()} (Accuracy: {acc*100:.1f}%)', fontsize=20)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.xticks(rotation=90, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()

print("\n--- ANALYSIS COMPLETE ---")