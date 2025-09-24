# KAGGLE NOTEBOOK - COPY EACH CELL SEPARATELY

## Cell 1: Setup and Imports

```python
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import shutil
from datetime import datetime

print("🚀 ASL Training on Kaggle")
print("=" * 50)
print(f"TensorFlow Version: {tf.__version__}")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Available: {len(gpus)} GPU(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✅ Mixed precision enabled")
    HAS_GPU = True
else:
    print("❌ No GPU detected")
    HAS_GPU = False

print("=" * 50)
```

## Cell 2: Find Dataset

```python
# Find dataset path
input_dir = Path('/kaggle/input')
dataset_path = None

if input_dir.exists():
    print("🔍 Searching for dataset...")
    for item in input_dir.iterdir():
        print(f"   Found: {item}")

        # Look for asl_alphabet_train folder (might be nested)
        possible_paths = [
            item / 'asl_alphabet_train',  # Direct path
            item,  # Root might be the dataset itself
        ]

        for path in possible_paths:
            if path.exists() and path.is_dir():
                # Check if this directory contains letter folders (A, B, C, etc.)
                subdirs = [d for d in path.iterdir() if d.is_dir()]
                letter_dirs = [d for d in subdirs if d.name in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' or d.name in ['space', 'del', 'nothing']]

                if len(letter_dirs) > 10:  # Should have at least 10 letter folders
                    dataset_path = str(path)
                    print(f"✅ Dataset found with {len(letter_dirs)} letter folders: {dataset_path}")
                    break

        if dataset_path:
            break

if not dataset_path:
    print("❌ Dataset not found!")
    print("Expected structure: dataset/A/, dataset/B/, dataset/C/, etc.")
else:
    print(f"📁 Using dataset: {dataset_path}")

    # Debug: Show what's actually in the dataset
    dataset_check = Path(dataset_path)
    print(f"\n🔍 Dataset contents:")
    for item in sorted(dataset_check.iterdir()):
        if item.is_dir():
            count = len(list(item.glob('*.jpg')) + list(item.glob('*.png')))
            print(f"  📁 {item.name}: {count} images")
        else:
            print(f"  📄 {item.name}")
```

## Cell 2.5: Debug Dataset Structure (Use this to check your dataset)

```python
# DEBUG: Check the actual structure of your dataset
input_dir = Path('/kaggle/input')

print("🔍 Full dataset structure analysis:")
print("=" * 50)

for dataset_folder in input_dir.iterdir():
    print(f"\n📁 Dataset folder: {dataset_folder.name}")
    print(f"   Path: {dataset_folder}")

    if dataset_folder.is_dir():
        # Check contents
        contents = list(dataset_folder.iterdir())
        print(f"   Contents ({len(contents)} items):")

        for item in sorted(contents)[:20]:  # Show first 20 items
            if item.is_dir():
                # Count images in this directory
                img_count = len(list(item.glob('*.jpg')) + list(item.glob('*.png')) + list(item.glob('*.jpeg')))
                print(f"     📁 {item.name}: {img_count} images")
            else:
                size_mb = item.stat().st_size / (1024*1024)
                print(f"     📄 {item.name}: {size_mb:.1f} MB")

        if len(contents) > 20:
            print(f"     ... and {len(contents) - 20} more items")

print("\n" + "=" * 50)
print("🎯 Looking for the correct dataset path...")

# Try to find the path with A, B, C folders
correct_path = None
for dataset_folder in input_dir.iterdir():
    if dataset_folder.is_dir():
        # Check this level
        subdirs = [d for d in dataset_folder.iterdir() if d.is_dir()]
        letter_folders = [d for d in subdirs if d.name in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']

        if len(letter_folders) >= 10:
            correct_path = str(dataset_folder)
            print(f"✅ Found dataset at: {correct_path}")
            print(f"   Letter folders: {sorted([d.name for d in letter_folders])}")
            break

        # Check one level deeper (nested structure)
        for subdir in subdirs:
            if subdir.is_dir():
                nested_dirs = [d for d in subdir.iterdir() if d.is_dir()]
                nested_letters = [d for d in nested_dirs if d.name in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']

                if len(nested_letters) >= 10:
                    correct_path = str(subdir)
                    print(f"✅ Found nested dataset at: {correct_path}")
                    print(f"   Letter folders: {sorted([d.name for d in nested_letters])}")
                    break

            if correct_path:
                break

if correct_path:
    print(f"\n🎯 Use this path for training: {correct_path}")
    dataset_path = correct_path
else:
    print("❌ Could not find dataset with A-Z folders")
    print("Please check your dataset upload")
```

```python
def prepare_dataset(input_dir, output_dir):
    print(f"📁 Preparing dataset from: {input_dir}")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all class directories - filter for actual letter/word folders
    all_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    # Filter for valid ASL classes (letters + special classes)
    valid_classes = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') | {'space', 'del', 'nothing'}
    class_dirs = [d for d in all_dirs if d.name in valid_classes or len(d.name) == 1]

    print(f"Found {len(all_dirs)} total directories, {len(class_dirs)} valid classes")
    print(f"Valid classes: {sorted([d.name for d in class_dirs])}")

    if len(class_dirs) == 0:
        print("❌ No valid ASL class folders found!")
        print("Expected folders: A, B, C, ..., Z, space, del, nothing")
        print("Available folders:", [d.name for d in all_dirs])
        return None

    splits = ['train', 'val', 'test']
    split_ratios = [0.7, 0.15, 0.15]

    for split in splits:
        (output_path / split).mkdir(exist_ok=True)

    total_files = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}

    for class_dir in class_dirs:
        print(f"Processing: {class_dir.name}")

        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        image_files.sort()

        if len(image_files) == 0:
            print(f"  ⚠️  No images found in {class_dir.name}")
            continue

        print(f"  Found {len(image_files)} images")
        total_files += len(image_files)

        n_files = len(image_files)
        train_end = int(n_files * split_ratios[0])
        val_end = train_end + int(n_files * split_ratios[1])

        for split in splits:
            (output_path / split / class_dir.name).mkdir(exist_ok=True)

        for i, img_file in enumerate(image_files):
            if i < train_end:
                split = 'train'
            elif i < val_end:
                split = 'val'
            else:
                split = 'test'

            dst = output_path / split / class_dir.name / img_file.name
            shutil.copy2(img_file, dst)
            split_counts[split] += 1

    print(f"✅ Dataset prepared: {total_files:,} total files")
    print(f"   Train: {split_counts['train']:,}")
    print(f"   Val: {split_counts['val']:,}")
    print(f"   Test: {split_counts['test']:,}")
    return str(output_path)

# Prepare the dataset - IMPORTANT: Output must be in /kaggle/working/ (writable)
prepared_dataset = prepare_dataset(dataset_path, '/kaggle/working/dataset')
```

## Cell 4: Create Dataset Loaders

```python
def create_dataset(data_dir, batch_size=32, img_size=(200, 200)):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    return dataset, dataset.class_names

def augment_dataset(dataset, is_training=True):
    data_augmentation = tf.keras.Sequential([
        layers.Rescaling(1./255),
    ])

    if is_training:
        data_augmentation.add(layers.RandomRotation(0.1))
        data_augmentation.add(layers.RandomZoom(0.1))
        data_augmentation.add(layers.RandomContrast(0.2))
        data_augmentation.add(layers.RandomBrightness(0.2))

    dataset = dataset.map(
        lambda x, y: (data_augmentation(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset.prefetch(tf.data.AUTOTUNE)

# Set parameters
BATCH_SIZE = 64 if HAS_GPU else 32
IMG_SIZE = (200, 200)
OUTPUT_DIR = '/kaggle/working/models'

print(f"Batch size: {BATCH_SIZE}")
print(f"Image size: {IMG_SIZE}")

# Load datasets
train_ds, class_names = create_dataset(f"{prepared_dataset}/train", BATCH_SIZE, IMG_SIZE)
val_ds, _ = create_dataset(f"{prepared_dataset}/val", BATCH_SIZE, IMG_SIZE)
test_ds, _ = create_dataset(f"{prepared_dataset}/test", BATCH_SIZE, IMG_SIZE)

print(f"✅ Loaded {len(class_names)} classes: {class_names}")

# Apply augmentation
train_ds = augment_dataset(train_ds, True)
val_ds = augment_dataset(val_ds, False)
test_ds = augment_dataset(test_ds, False)

print("✅ Data augmentation applied")
```

## Cell 5: Create Model

```python
def create_model(num_classes, img_size=(200, 200)):
    base_model = MobileNetV3Large(
        input_shape=(*img_size, 3),
        alpha=1.0,
        minimalistic=False,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model, base_model

# Calculate class weights
def calculate_class_weights(dataset, class_names):
    class_counts = np.zeros(len(class_names))
    total_samples = 0

    for images, labels in dataset:
        label_indices = tf.argmax(labels, axis=1)
        for idx in label_indices:
            class_counts[idx.numpy()] += 1
        total_samples += len(labels)

    class_weights = {}
    for i, count in enumerate(class_counts):
        if count > 0:
            class_weights[i] = total_samples / (len(class_names) * count)
        else:
            class_weights[i] = 1.0

    return class_weights

# Create model and calculate weights
model, base_model = create_model(len(class_names), IMG_SIZE)
class_weights = calculate_class_weights(train_ds, class_names)

print(f"✅ Model created with {model.count_params():,} parameters")
print(f"✅ Class weights calculated")
```

## Cell 6: Train Model - Phase 1

```python
# Setup output directory
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

print("🎯 Phase 1: Training classifier head")

# Phase 1 callbacks
callbacks_phase1 = [
    ModelCheckpoint(
        str(output_path / 'best_model_phase1.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

# Train phase 1 (15 epochs)
history_phase1 = model.fit(
    train_ds,
    epochs=15,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=callbacks_phase1,
    verbose=1
)

print("✅ Phase 1 completed")
```

## Cell 7: Train Model - Phase 2

```python
print("🔥 Phase 2: Fine-tuning entire model")

# Unfreeze base model
base_model.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00002),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Phase 2 callbacks
callbacks_phase2 = [
    ModelCheckpoint(
        str(output_path / 'best_model_final.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.000001,
        verbose=1
    )
]

# Train phase 2 (15 more epochs) - FIXED with proper epoch counting
try:
    print("Starting Phase 2 training...")
    history_phase2 = model.fit(
        train_ds,
        epochs=30,  # Total epochs (15 + 15)
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_phase2,
        verbose=1,
        initial_epoch=15  # Start from epoch 15, train to epoch 30
    )
    print("✅ Phase 2 completed successfully")

    # Check if training actually happened
    if len(history_phase2.history) == 0 or len(history_phase2.history.get('loss', [])) == 0:
        print("⚠️  Phase 2 didn't train any epochs (check epoch settings)")
        raise ValueError("No epochs trained in Phase 2")

except Exception as e:
    print(f"⚠️  Phase 2 training failed: {e}")
    print("Creating empty history for phase 2...")
    # Create empty history object
    class EmptyHistory:
        def __init__(self):
            self.history = {}
    history_phase2 = EmptyHistory()

# Combine histories - Fix for empty Phase 2 history
combined_history = {}

# Debug: Print available keys
print("Phase 1 history keys:", list(history_phase1.history.keys()))
print("Phase 2 history keys:", list(history_phase2.history.keys()))

# Check if Phase 2 training actually happened
if len(history_phase2.history) == 0:
    print("⚠️  Phase 2 training failed - using only Phase 1 history")
    combined_history = history_phase1.history.copy()
else:
    # Normal case - combine both phases
    common_keys = set(history_phase1.history.keys()) & set(history_phase2.history.keys())
    print(f"Common keys: {common_keys}")

    for key in common_keys:
        combined_history[key] = history_phase1.history[key] + history_phase2.history[key]

    # Add any missing keys from phase 1
    for key in history_phase1.history.keys():
        if key not in combined_history:
            if 'loss' in history_phase2.history:
                combined_history[key] = history_phase1.history[key] + [None] * len(history_phase2.history['loss'])
            else:
                combined_history[key] = history_phase1.history[key]
            print(f"⚠️  Added {key} from phase 1 only")

print("✅ Training completed!")
```

## Cell 8: Evaluate and Export

```python
# Evaluate on test set
print("📊 Evaluating model...")

predictions = []
true_labels = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    predictions.extend(np.argmax(preds, axis=1))
    true_labels.extend(np.argmax(labels, axis=1))

test_accuracy = np.mean(np.array(predictions) == np.array(true_labels))
print(f"✅ Test Accuracy: {test_accuracy:.4f}")

# Export models
print("📦 Exporting models...")

# Save Keras model
model.save(output_path / 'model.keras')

# Save labels
with open(output_path / 'training_set_labels.txt', 'w') as f:
    for label in class_names:
        f.write(f"{label}\\n")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(output_path / 'model.tflite', 'wb') as f:
    f.write(tflite_model)

# Save training history
with open(output_path / 'training_history.json', 'w') as f:
    json.dump(combined_history, f, indent=2)

print("✅ All models exported!")

# Final summary
print("\\n" + "=" * 60)
print("🎉 TRAINING COMPLETED!")
print(f"📊 Test Accuracy: {test_accuracy:.4f}")
print(f"📁 Models saved to: {OUTPUT_DIR}")
print("=" * 60)
```

## Cell 9: Plot Results

```python
# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0,0].plot(combined_history['accuracy'], label='Training')
axes[0,0].plot(combined_history['val_accuracy'], label='Validation')
axes[0,0].set_title('Accuracy')
axes[0,0].legend()
axes[0,0].grid(True)

# Loss
axes[0,1].plot(combined_history['loss'], label='Training')
axes[0,1].plot(combined_history['val_loss'], label='Validation')
axes[0,1].set_title('Loss')
axes[0,1].legend()
axes[0,1].grid(True)

# Final metrics
final_acc = combined_history['val_accuracy'][-1]
final_loss = combined_history['val_loss'][-1]
axes[1,0].text(0.5, 0.5, f'Final Validation\\nAccuracy: {final_acc:.4f}',
               ha='center', va='center', transform=axes[1,0].transAxes, fontsize=14)
axes[1,0].set_title('Final Accuracy')

axes[1,1].text(0.5, 0.5, f'Test\\nAccuracy: {test_accuracy:.4f}',
               ha='center', va='center', transform=axes[1,1].transAxes, fontsize=14)
axes[1,1].set_title('Test Performance')

plt.tight_layout()
plt.savefig(output_path / 'training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# List generated files
print("📁 Generated files:")
for file in output_path.rglob('*'):
    if file.is_file():
        size_mb = file.stat().st_size / (1024*1024)
        print(f"   {file.name}: {size_mb:.1f} MB")
```
