import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ===========================================
# CONFIGURATION
# ===========================================
# 1. Path to the ORIGINAL balanced dataset (The "Final Exam" for both models)
TEST_DIR = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\TeacherModel\plantvillage_whole_balanced"

# 2. Paths to the trained models
PATH_BASELINE = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\StudentModel\student_baseline_standard.h5"
PATH_CURRICULUM = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\StudentModel\student_final_curriculum_MC_DROPOUT.h5"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 15

# ===========================================
# 1. SETUP DATA GENERATOR
# ===========================================
print("--- Setting up Data Generators ---")
if not os.path.exists(TEST_DIR):
    print(f"❌ Error: Dataset not found at {TEST_DIR}")
    sys.exit()

# We use the 'validation' split of the original dataset as our Test Set
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False
)

class_names = list(test_gen.class_indices.keys())

# ===========================================
# 2. EVALUATION HELPER FUNCTION
# ===========================================
def evaluate_model(model_path, name):
    print(f"\n{'='*40}")
    print(f"🔍 EVALUATING: {name}")
    print(f"{'='*40}")
    
    if not os.path.exists(model_path):
        print(f"⚠️ Warning: Model file not found: {model_path}")
        return None

    # Load Model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ {name} loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        return None

    # 1. Run Predictions
    print("   Running inference on validation set...")
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    # 2. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"   📊 Accuracy: {acc*100:.2f}%")
    
    # 3. Detailed Report
    print("\n   --- Classification Report ---")
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 4. Return results for plotting later
    return {
        'name': name,
        'accuracy': acc,
        'y_true': y_true,
        'y_pred': y_pred,
        'report': report_dict,
        'cm': confusion_matrix(y_true, y_pred)
    }

# ===========================================
# 3. RUN COMPARISON
# ===========================================
results_baseline = evaluate_model(PATH_BASELINE, "Baseline Model")
# Clear memory to be safe
tf.keras.backend.clear_session()
results_curriculum = evaluate_model(PATH_CURRICULUM, "Curriculum Model")

# ===========================================
# 4. VISUALIZATION & CONCLUSION
# ===========================================
if results_baseline and results_curriculum:
    print("\n\n")
    print("="*60)
    print("🏆 FINAL RESULTS SUMMARY")
    print("="*60)
    
    acc_base = results_baseline['accuracy'] * 100
    acc_curr = results_curriculum['accuracy'] * 100
    diff = acc_curr - acc_base
    
    print(f"1. Baseline Accuracy:   {acc_base:.2f}%")
    print(f"2. Curriculum Accuracy: {acc_curr:.2f}%")
    print("-" * 30)
    
    if diff > 0:
        print(f"✅ CONCLUSION: Curriculum Learning IMPROVED accuracy by +{diff:.2f}%")
    else:
        print(f"❌ CONCLUSION: Curriculum Learning did NOT improve accuracy ({diff:.2f}%)")

    # --- PLOT 1: Confusion Matrices Side-by-Side ---
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Baseline CM
    sns.heatmap(results_baseline['cm'], annot=True, fmt='d', cmap='Reds', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(f"Baseline Model (Acc: {acc_base:.1f}%)", fontsize=16)
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].tick_params(axis='x', rotation=90)

    # Curriculum CM
    sns.heatmap(results_curriculum['cm'], annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title(f"Curriculum Model (Acc: {acc_curr:.1f}%)", fontsize=16)
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    axes[1].tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Class-wise Improvement ---
    # Which specific diseases got better?
    base_f1 = [results_baseline['report'][c]['f1-score'] for c in class_names]
    curr_f1 = [results_curriculum['report'][c]['f1-score'] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, base_f1, width, label='Baseline', color='salmon')
    plt.bar(x + width/2, curr_f1, width, label='Curriculum', color='lightgreen')
    
    plt.ylabel('F1 Score')
    plt.title('Class-wise Performance Comparison')
    plt.xticks(x, class_names, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

else:
    print("\nCould not run comparison. Please ensure both .h5 files exist.")