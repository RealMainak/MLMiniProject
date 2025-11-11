import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# ===========================================
# USER SETTINGS
# ===========================================
dataset_dir = r"C:\Users\bosem\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\plantvillage"

# CHANGED: Corrected list to match the actual folder names in that dataset
# Note the inconsistent underscores (some have 1, some have 2)

selected_classes = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

img_size = (224, 224)
batch_size = 16
val_split = 0.2
epochs = 15

# ===========================================
# STEP 1: FILTER CLASSES
# ===========================================
# This code block works as-is and will use the new 'selected_classes' list
available_classes = [d for d in os.listdir(dataset_dir) if d in selected_classes]
print(f"✅ Using {len(available_classes)} classes:", available_classes)

# ===========================================
# STEP 2: DATA GENERATORS (with validation split)
# ===========================================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=val_split
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    classes=available_classes,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    classes=available_classes,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ===========================================
# STEP 3: PRETRAINED MODEL (EfficientNetB0)
# ===========================================
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False  # Freeze pretrained layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    # The output layer automatically adjusts to the correct number of classes
    layers.Dense(len(available_classes), activation='softmax')
])

model.compile(
    # Increased learning rate as requested
    optimizer=Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===========================================
# STEP 4: TRAIN MODEL
# ===========================================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

# ===========================================
# STEP 5: EVALUATE AND SAVE
# ===========================================
loss, accuracy = model.evaluate(val_gen)
print(f"\n✅ Final Validation Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Final Validation Loss: {loss:.4f}")

# Updated save file name
model.save_weights("tomato_disease_effnetb0_split_weights.h5")
print("💾 Model saved successfully!")

# ===========================================
# STEP 6: PLOT ACCURACY & LOSS CURVES
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