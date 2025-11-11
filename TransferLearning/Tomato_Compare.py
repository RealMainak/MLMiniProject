import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import MobileNetV2, ResNet50V2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.optimizers import Adam
import os
import sys

# --- ADDED FOR CONFUSION MATRIX ---
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# ------------------------------------


# ===========================================
# SETTINGS
# (These must match the training script)
# ===========================================
balanced_dir = "tomato_balanced_dataset" 
selected_classes = [
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", 
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus"
]
img_size = (224, 224)
batch_size = 16
val_split = 0.2

# ===========================================
# HELPER FUNCTION TO BUILD MODEL
# ===========================================
def build_model(model_name, num_classes):
    """
    Builds the exact model architecture used during training.
    This is CRITICAL for loading weights correctly.
    """
    inputs = layers.Input(shape=img_size + (3,)) 
    
    if model_name == 'mobilenet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
        unfreeze_last_n_layers = 50
    elif model_name == 'resnet':
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
        unfreeze_last_n_layers = 50
    else:
        raise ValueError("Unknown model_name")
        
    # --- Replicate the exact fine-tuning setup ---
    base_model.trainable = True
    freeze_until_layer = len(base_model.layers) - unfreeze_last_n_layers

    for layer in base_model.layers[:freeze_until_layer]:
        layer.trainable = False
    
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    # --- End replication ---
    
    x = base_model(inputs, training=False) # Use training=False for evaluation
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    
    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- NEW HELPER FUNCTION FOR PLOTTING CM ---
def plot_confusion_matrix(model, validation_generator, model_name):
    """
    Generates and plots a confusion matrix for a given model
    and validation generator.
    """
    print(f"\n--- Generating Confusion Matrix for {model_name} ---")
    
    # 1. Get True Labels
    validation_generator.reset()
    true_labels = validation_generator.classes
    
    # 2. Get Predicted Labels
    predictions = model.predict(validation_generator, 
                                steps=len(validation_generator), 
                                verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # 3. Get Class Names
    class_names = list(validation_generator.class_indices.keys())
    
    # 4. Compute Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # 5. Plot Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {model_name.upper()}', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    print(f"Confusion Matrix plot for {model_name} created.")
# -----------------------------------------------


# ===========================================
# SCRIPT EXECUTION
# ===========================================
print("--- Starting Model Comparison ---")
results = {}
num_classes = len(selected_classes)

# --- 1. Evaluate MobileNetV2 ---
print("\nLoading and evaluating MobileNetV2...")
try:
    # Create the validation generator with MobileNet preprocessing
    mnet_datagen = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess,
        validation_split=val_split
    )
    mnet_val_gen = mnet_datagen.flow_from_directory(
        balanced_dir, target_size=img_size, batch_size=batch_size,
        classes=selected_classes, class_mode='categorical',
        subset='validation', shuffle=False
    )
    
    # Build model and load weights
    model_mnet = build_model('mobilenet', num_classes)
    weights_mnet = "tomato_disease_mobilenet_2stage_weights.h5"
    if not os.path.exists(weights_mnet):
        raise FileNotFoundError(f"Missing file: {weights_mnet}")
        
    model_mnet.load_weights(weights_mnet)
    print("MobileNetV2 weights loaded.")
    
    # Evaluate
    loss, acc = model_mnet.evaluate(mnet_val_gen)
    results['MobileNetV2'] = {'loss': loss, 'accuracy': acc}

    # --- ADDED: PLOT CM FOR MNET ---
    plot_confusion_matrix(model_mnet, mnet_val_gen, "MobileNetV2")
    # -------------------------------

except Exception as e:
    print(f"Error evaluating MobileNetV2: {e}")
    print("Please make sure 'tomato_disease_mobilenet_2stage_weights.h5' exists.")

# --- 2. Evaluate ResNet50V2 ---
print("\nLoading and evaluating ResNet50V2...")
try:
    # Create the validation generator with ResNet preprocessing
    resnet_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split
    )
    resnet_val_gen = resnet_datagen.flow_from_directory(
        balanced_dir, target_size=img_size, batch_size=batch_size,
        classes=selected_classes, class_mode='categorical',
        subset='validation', shuffle=False
    )
    
    # Build model and load weights
    model_resnet = build_model('resnet', num_classes)
    weights_resnet = "tomato_disease_resnet_2stage_weights.h5"
    if not os.path.exists(weights_resnet):
        raise FileNotFoundError(f"Missing file: {weights_resnet}")
        
    model_resnet.load_weights(weights_resnet)
    print("ResNet50V2 weights loaded.")
    
    # Evaluate
    loss, acc = model_resnet.evaluate(resnet_val_gen)
    results['ResNet50V2'] = {'loss': loss, 'accuracy': acc}

    # --- ADDED: PLOT CM FOR RESNET ---
    plot_confusion_matrix(model_resnet, resnet_val_gen, "ResNet50V2")
    # ---------------------------------

except Exception as e:
    print(f"Error evaluating ResNet50V2: {e}")
    print("Please make sure 'tomato_disease_resnet_2stage_weights.h5' exists.")

# --- 3. Print Final Comparison ---
print("\n\n--- FINAL PERFORMANCE COMPARISON ---")
print("----------------------------------------")
print(f"| Model       | Val. Accuracy | Val. Loss |")
print(f"|-------------|---------------|-----------|")

for model_name, metrics in results.items():
    acc_str = f"{metrics['accuracy'] * 100:.2f}%"
    loss_str = f"{metrics['loss']:.4f}"
    print(f"| {model_name:<11} | {acc_str:<13} | {loss_str:<9} |")

print("----------------------------------------")


# --- ADDED: SHOW ALL PLOTS AT THE END ---
if results: # Only show plots if at least one model was evaluated
    print("\nDisplaying plots...")
    plt.show()
else:
    print("\nNo models were successfully evaluated. No plots to display.")

print("\n--- SCRIPT FINISHED ---")