import tensorflow as tf
import os

# 1. Load your trained model
# Replace with your actual model file name
model_path = r"E:\Study Materials\Projects\Curriculum Learning\MultiClass\StudentModel\student_final_curriculum_STANDARD.h5"

if not os.path.exists(model_path):
    print(f"❌ File not found: {model_path}")
else:
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully.")

    # --- METHOD 1: THEORETICAL CALCULATION ---
    # In standard TensorFlow, every parameter is a 32-bit float (4 bytes).
    total_params = model.count_params()
    theoretical_size_bytes = total_params * 4
    theoretical_size_mb = theoretical_size_bytes / (1024 * 1024)

    print(f"\n📊 [Theoretical] Parameter Count: {total_params:,}")
    print(f"📊 [Theoretical] Estimated Size (Float32): {theoretical_size_mb:.2f} MB")

    # --- METHOD 2: ACTUAL DISK SIZE ---
    # This includes the architecture definition and metadata, so it's slightly larger.
    file_size_bytes = os.path.getsize(model_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(f"\n💾 [Actual] Disk File Size (.h5): {file_size_mb:.2f} MB")

    # --- JUDGMENT ---
    limit_mb = 2.5
    if file_size_mb < limit_mb:
        print(f"\n✅ SUCCESS: The model is under {limit_mb} MB.")
    else:
        print(f"\n⚠️ WARNING: The model is {file_size_mb:.2f} MB.")
        print("   Don't panic! TinyML models often use 'Quantization' (INT8) for deployment.")
        print(f"   Estimated Quantized Size (INT8): {file_size_mb / 4:.2f} MB")