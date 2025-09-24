# Kaggle Notebook Code Cells

# Copy these cells into your Kaggle notebook in order

## Cell 1: System Setup and GPU Check

```python
import os
import sys

# Check GPU availability
import tensorflow as tf
print("🔧 System Information")
print("=" * 40)
print(f"TensorFlow Version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Available: {len(gpus)} GPU(s)")
    for gpu in gpus:
        print(f"   {gpu}")
else:
    print("❌ No GPU - Enable GPU accelerator in Kaggle!")

print("\n📁 Available Datasets:")
input_dir = '/kaggle/input'
if os.path.exists(input_dir):
    for item in os.listdir(input_dir):
        print(f"   {item}")
```

## Cell 2: Install Dependencies (if needed)

```python
# Usually not needed as Kaggle has most ML libraries pre-installed
# Uncomment if you need specific versions

# !pip install tensorflow==2.17.0
# !pip install scikit-learn matplotlib seaborn
```

## Cell 3: Upload and Run Training Script

```python
# Method 1: Copy the training script directly into a cell
# OR
# Method 2: Upload as a file and run it

# Create the training script
training_script = '''
# [PASTE THE ENTIRE kaggle_train.py CONTENT HERE]
'''

# Save to file
with open('/kaggle/working/train_asl.py', 'w') as f:
    f.write(training_script)

print("✅ Training script saved to /kaggle/working/train_asl.py")
```

## Cell 4: Run Training

```python
# Execute the training script
exec(open('/kaggle/working/train_asl.py').read())

# OR run as subprocess to see live output
# import subprocess
# result = subprocess.run([sys.executable, '/kaggle/working/train_asl.py'],
#                        capture_output=True, text=True)
# print(result.stdout)
```

## Cell 5: Monitor Progress (Optional - Run in separate cell)

```python
import time
import os
from pathlib import Path

models_dir = Path('/kaggle/working/models')

while True:
    if models_dir.exists():
        files = list(models_dir.rglob('*'))
        print(f"⏱️  Files created: {len(files)}")

        # Check for specific files
        if (models_dir / 'model.keras').exists():
            print("✅ Keras model saved")
        if (models_dir / 'model.tflite').exists():
            print("✅ TFLite model saved")
        if (models_dir / 'training_history.json').exists():
            print("✅ Training history saved")

    time.sleep(60)  # Check every minute
```

## Cell 6: View Results

```python
import json
import matplotlib.pyplot as plt
from pathlib import Path

models_dir = Path('/kaggle/working/models')

# Load and display training history
if (models_dir / 'training_history.json').exists():
    with open(models_dir / 'training_history.json', 'r') as f:
        history = json.load(f)

    print(f"Training completed with {len(history['accuracy'])} epochs")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")

# Load evaluation results
if (models_dir / 'evaluation_report.json').exists():
    with open(models_dir / 'evaluation_report.json', 'r') as f:
        results = json.load(f)

    print(f"Test accuracy: {results['test_accuracy']:.4f}")

# List all generated files
print("\n📁 Generated Files:")
if models_dir.exists():
    for file in models_dir.rglob('*'):
        if file.is_file():
            size_mb = file.stat().st_size / (1024*1024)
            print(f"   {file.name}: {size_mb:.1f} MB")
```

## Cell 7: Test Model (Optional)

```python
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Load the trained model
model_path = '/kaggle/working/models/model.keras'
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)

    # Load class names
    with open('/kaggle/working/models/training_set_labels.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    print(f"✅ Model loaded with {len(class_names)} classes")
    print(f"Model input shape: {model.input_shape}")

    # Test with a sample from test dataset
    test_dir = '/kaggle/working/dataset/test'
    if os.path.exists(test_dir):
        # Pick a random test image
        import random
        class_dir = random.choice([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
        class_path = os.path.join(test_dir, class_dir)
        img_file = random.choice(os.listdir(class_path))
        img_path = os.path.join(class_path, img_file)

        # Load and preprocess image
        img = keras.utils.load_img(img_path, target_size=(200, 200))
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        # Display result
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f'Predicted: {predicted_class} (Confidence: {confidence:.3f})\nActual: {class_dir}')
        plt.axis('off')
        plt.show()

        print(f"Actual: {class_dir}")
        print(f"Predicted: {predicted_class}")
        print(f"Confidence: {confidence:.3f}")
```

## Cell 8: Create Submission Package

```python
import zipfile
from pathlib import Path

# Create a zip file with all important files
models_dir = Path('/kaggle/working/models')
zip_path = '/kaggle/working/asl_trained_models.zip'

if models_dir.exists():
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in models_dir.rglob('*'):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(models_dir.parent))

    print(f"✅ Models packaged in: {zip_path}")
    print(f"📦 Package size: {os.path.getsize(zip_path) / (1024*1024):.1f} MB")
    print("\n📥 Download this file to get all your trained models!")
else:
    print("❌ Models directory not found")

# List final deliverables
print("\n📋 Final Deliverables:")
important_files = [
    'model.keras',
    'model.tflite',
    'training_set_labels.txt',
    'frozen_model.pb',
    'evaluation_report.json',
    'training_history.json'
]

for filename in important_files:
    file_path = models_dir / filename
    if file_path.exists():
        print(f"✅ {filename}")
    else:
        print(f"❌ {filename} - Missing")
```
