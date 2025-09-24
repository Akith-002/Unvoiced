"""
ASL Sign Language Classification Training for Kaggle
Optimized for Kaggle's GPU environment with 16GB GPU memory.

This script trains a MobileNetV3Large model on the ASL alphabet dataset.
Expected training time: 2-3 hours on Kaggle GPU vs 8-12 hours on CPU.

Usage in Kaggle Notebook:
1. Add your ASL dataset as a data source
2. Enable GPU accelerator 
3. Run this script
4. Download the trained models from /kaggle/working/models/
"""

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

# Print system info
print("🚀 ASL Training on Kaggle")
print("=" * 50)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Available: {len(gpus)} GPU(s)")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu}")
    
    # Configure GPU memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth configured")
    except RuntimeError as e:
        print(f"GPU config error: {e}")
    
    # Enable mixed precision for better GPU utilization
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✅ Mixed precision enabled")
else:
    print("❌ No GPU detected - using CPU")

print("=" * 50)

def find_dataset_path():
    """Find the dataset path in Kaggle environment"""
    # Common Kaggle dataset paths
    possible_paths = [
        '/kaggle/input/asl-dataset/asl_alphabet_train',
        '/kaggle/input/asl-alphabet/asl_alphabet_train', 
        '/kaggle/input/asl-alphabet-train/asl_alphabet_train',
        '/kaggle/input/sign-language/asl_alphabet_train',
        '/kaggle/input/asl-data/asl_alphabet_train'
    ]
    
    # Check Kaggle input directory
    input_dir = Path('/kaggle/input')
    if input_dir.exists():
        print("🔍 Searching for dataset in Kaggle input...")
        for item in input_dir.iterdir():
            print(f"   Found: {item}")
            # Look for asl_alphabet_train folder
            asl_path = item / 'asl_alphabet_train'
            if asl_path.exists():
                print(f"✅ Dataset found: {asl_path}")
                return str(asl_path)
    
    # Check predefined paths
    for path in possible_paths:
        if Path(path).exists():
            print(f"✅ Dataset found: {path}")
            return path
    
    print("❌ Dataset not found. Please check your Kaggle dataset setup.")
    return None

def prepare_dataset(input_dir, output_dir):
    """Prepare dataset by splitting into train/val/test"""
    print(f"📁 Preparing dataset from: {input_dir}")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    
    # Create output structure
    splits = ['train', 'val', 'test']
    split_ratios = [0.7, 0.15, 0.15]  # 70% train, 15% val, 15% test
    
    for split in splits:
        (output_path / split).mkdir(exist_ok=True)
    
    total_files = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    # Process each class
    for class_dir in class_dirs:
        print(f"Processing class: {class_dir.name}")
        
        # Get all images in this class
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        image_files.sort()  # Ensure consistent ordering
        
        if len(image_files) == 0:
            print(f"⚠️  No images found in {class_dir}")
            continue
        
        total_files += len(image_files)
        
        # Calculate split indices
        n_files = len(image_files)
        train_end = int(n_files * split_ratios[0])
        val_end = train_end + int(n_files * split_ratios[1])
        
        # Create class directories in each split
        for split in splits:
            split_class_dir = output_path / split / class_dir.name
            split_class_dir.mkdir(exist_ok=True)
        
        # Copy files to appropriate splits
        for i, img_file in enumerate(image_files):
            if i < train_end:
                split = 'train'
            elif i < val_end:
                split = 'val'
            else:
                split = 'test'
            
            dst_path = output_path / split / class_dir.name / img_file.name
            shutil.copy2(img_file, dst_path)
            split_counts[split] += 1
    
    print(f"✅ Dataset prepared:")
    print(f"   Total files: {total_files:,}")
    print(f"   Train: {split_counts['train']:,} ({split_counts['train']/total_files*100:.1f}%)")
    print(f"   Val: {split_counts['val']:,} ({split_counts['val']/total_files*100:.1f}%)")
    print(f"   Test: {split_counts['test']:,} ({split_counts['test']/total_files*100:.1f}%)")
    
    return str(output_path)

def create_dataset(data_dir, batch_size=32, img_size=(200, 200), validation_split=0.0, subset=None):
    """Create tf.data.Dataset from directory structure"""
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset=subset,
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    return dataset, dataset.class_names

def augment_dataset(dataset, is_training=True):
    """Apply data augmentation for training"""
    
    # Build augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        layers.Rescaling(1./255),  # Normalize to [0,1]
    ])
    
    if is_training:
        # Add augmentations only for training
        data_augmentation.add(layers.RandomRotation(0.1))  # ±15 degrees
        data_augmentation.add(layers.RandomZoom(0.1))      # ±10% zoom
        data_augmentation.add(layers.RandomContrast(0.2))   # ±20% contrast
        data_augmentation.add(layers.RandomBrightness(0.2)) # ±20% brightness
        # Note: No horizontal flip for ASL as it would change meaning
    
    # Apply augmentation
    dataset = dataset.map(
        lambda x, y: (data_augmentation(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Optimize dataset performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_model(num_classes, img_size=(200, 200)):
    """Create MobileNetV3Large model optimized for mobile deployment"""
    
    # Load pre-trained MobileNetV3Large
    base_model = MobileNetV3Large(
        input_shape=(*img_size, 3),
        alpha=1.0,
        minimalistic=False,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    inputs = keras.Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile with mixed precision compatible loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    return model, base_model

def calculate_class_weights(dataset, class_names):
    """Calculate class weights for balanced training"""
    print("📊 Calculating class weights...")
    
    # Count samples per class
    class_counts = np.zeros(len(class_names))
    total_samples = 0
    
    for images, labels in dataset:
        label_indices = tf.argmax(labels, axis=1)
        for idx in label_indices:
            class_counts[idx.numpy()] += 1
        total_samples += len(labels)
    
    # Calculate weights
    class_weights = {}
    for i, count in enumerate(class_counts):
        if count > 0:
            class_weights[i] = total_samples / (len(class_names) * count)
        else:
            class_weights[i] = 1.0
    
    print(f"Class distribution:")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        print(f"  {name}: {int(count):,} samples (weight: {class_weights[i]:.3f})")
    
    return class_weights

def train_model(model, base_model, train_ds, val_ds, class_weights, epochs=30, output_dir='models'):
    """Train the model with two-phase approach"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🎯 Starting Phase 1: Train classifier head (frozen base)")
    
    # Phase 1: Train only the classifier head
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
    
    # Train phase 1
    history_phase1 = model.fit(
        train_ds,
        epochs=epochs//2,  # Half epochs for phase 1
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    print("\n🔥 Starting Phase 2: Fine-tune entire model")
    
    # Phase 2: Unfreeze and fine-tune
    base_model.trainable = True
    
    # Use lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001/5),  # Much lower LR
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
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
    
    # Train phase 2
    history_phase2 = model.fit(
        train_ds,
        epochs=epochs//2,  # Remaining epochs for phase 2
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_phase2,
        verbose=1,
        initial_epoch=len(history_phase1.history['loss'])
    )
    
    # Combine histories
    combined_history = {}
    for key in history_phase1.history.keys():
        combined_history[key] = history_phase1.history[key] + history_phase2.history[key]
    
    # Save training history
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(combined_history, f, indent=2)
    
    print("✅ Training completed!")
    return model, combined_history

def evaluate_model(model, test_ds, class_names, output_dir='models'):
    """Evaluate model on test set"""
    print("📊 Evaluating model...")
    
    output_path = Path(output_dir)
    
    # Get predictions
    predictions = []
    true_labels = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        predictions.extend(np.argmax(preds, axis=1))
        true_labels.extend(np.argmax(labels, axis=1))
    
    # Calculate metrics
    test_accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    
    # Classification report
    report = classification_report(
        true_labels, predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Save detailed results
    with open(output_path / 'evaluation_report.json', 'w') as f:
        json.dump({
            'test_accuracy': float(test_accuracy),
            'classification_report': report
        }, f, indent=2)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Test Accuracy: {test_accuracy:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    print(f"📄 Detailed report saved to: {output_path / 'evaluation_report.json'}")
    
    return {'test_accuracy': test_accuracy, 'report': report}

def export_models(model, class_names, output_dir='models'):
    """Export model in multiple formats for deployment"""
    print("📦 Exporting models...")
    
    output_path = Path(output_dir)
    
    # 1. Save Keras model
    model.save(output_path / 'model.keras')
    print("✅ Keras model saved")
    
    # 2. Save class labels
    with open(output_path / 'training_set_labels.txt', 'w') as f:
        for label in class_names:
            f.write(f"{label}\n")
    print("✅ Class labels saved")
    
    # 3. Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Enable mixed precision for mobile
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open(output_path / 'model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("✅ TensorFlow Lite model saved")
    
    # 4. Create frozen graph (compatible with original code)
    try:
        # Convert to concrete function
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
        )
        
        # Get frozen ConcreteFunction
        frozen_func = tf.graph_util.convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        
        # Save frozen graph
        tf.io.write_graph(
            graph_or_graph_def=frozen_func.graph,
            logdir=str(output_path),
            name='frozen_model.pb',
            as_text=False
        )
        print("✅ Frozen graph saved")
        
    except Exception as e:
        print(f"⚠️  Frozen graph export failed: {e}")
    
    # 5. Create model summary
    with open(output_path / 'model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print("✅ Model summary saved")
    
    print(f"📦 All models exported to: {output_path}")

def plot_training_history(history, output_dir='models'):
    """Plot and save training history"""
    output_path = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy plot
    axes[0,0].plot(history['accuracy'], label='Training Accuracy')
    axes[0,0].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0,0].set_title('Model Accuracy')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Loss plot
    axes[0,1].plot(history['loss'], label='Training Loss')
    axes[0,1].plot(history['val_loss'], label='Validation Loss')
    axes[0,1].set_title('Model Loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Learning rate (if available)
    if 'lr' in history:
        axes[1,0].plot(history['lr'])
        axes[1,0].set_title('Learning Rate')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Learning Rate')
        axes[1,0].set_yscale('log')
        axes[1,0].grid(True)
    else:
        axes[1,0].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
    
    # Final metrics
    final_acc = history['val_accuracy'][-1]
    final_loss = history['val_loss'][-1]
    axes[1,1].text(0.5, 0.7, f'Final Validation Accuracy\n{final_acc:.4f}', 
                  ha='center', va='center', transform=axes[1,1].transAxes, fontsize=14)
    axes[1,1].text(0.5, 0.3, f'Final Validation Loss\n{final_loss:.4f}', 
                  ha='center', va='center', transform=axes[1,1].transAxes, fontsize=14)
    axes[1,1].set_title('Final Metrics')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training pipeline for Kaggle"""
    
    print("🚀 Starting ASL Training Pipeline on Kaggle")
    print("=" * 60)
    
    # Find dataset
    input_dataset_path = find_dataset_path()
    if not input_dataset_path:
        print("❌ Dataset not found. Please ensure your dataset is uploaded to Kaggle.")
        return
    
    # Prepare dataset (split into train/val/test)
    dataset_path = prepare_dataset(input_dataset_path, '/kaggle/working/dataset')
    
    # Training parameters - optimized for Kaggle GPU
    BATCH_SIZE = 64 if gpus else 32  # Larger batch size for GPU
    EPOCHS = 30  # More epochs for better results
    IMG_SIZE = (200, 200)
    OUTPUT_DIR = '/kaggle/working/models'
    
    print(f"🔧 Training Configuration:")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Image Size: {IMG_SIZE}")
    print(f"   Output Dir: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Load datasets
    print("📂 Loading datasets...")
    train_ds, class_names = create_dataset(
        f"{dataset_path}/train", 
        batch_size=BATCH_SIZE, 
        img_size=IMG_SIZE
    )
    val_ds, _ = create_dataset(
        f"{dataset_path}/val", 
        batch_size=BATCH_SIZE, 
        img_size=IMG_SIZE
    )
    test_ds, _ = create_dataset(
        f"{dataset_path}/test", 
        batch_size=BATCH_SIZE, 
        img_size=IMG_SIZE
    )
    
    print(f"✅ Found {len(class_names)} classes: {class_names}")
    
    # Apply augmentation
    print("🎨 Applying data augmentation...")
    train_ds = augment_dataset(train_ds, is_training=True)
    val_ds = augment_dataset(val_ds, is_training=False)
    test_ds = augment_dataset(test_ds, is_training=False)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_ds, class_names)
    
    # Create model
    print("🏗️  Creating model...")
    model, base_model = create_model(len(class_names), IMG_SIZE)
    print(f"✅ Model created. Total parameters: {model.count_params():,}")
    
    # Train model
    print("🎯 Starting training...")
    start_time = datetime.now()
    model, history = train_model(
        model, base_model, train_ds, val_ds, class_weights, 
        epochs=EPOCHS, output_dir=OUTPUT_DIR
    )
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"⏱️  Training completed in: {training_duration}")
    
    # Evaluate model
    print("📊 Evaluating model...")
    evaluation_results = evaluate_model(model, test_ds, class_names, OUTPUT_DIR)
    
    # Plot training history
    plot_training_history(history, OUTPUT_DIR)
    
    # Export models
    export_models(model, class_names, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Test Accuracy: {evaluation_results['test_accuracy']:.4f}")
    print(f"⏱️  Training Time: {training_duration}")
    print(f"📁 Models saved to: {OUTPUT_DIR}")
    print(f"📱 TFLite model: {OUTPUT_DIR}/model.tflite")
    print(f"🏷️  Labels file: {OUTPUT_DIR}/training_set_labels.txt")
    print("=" * 60)
    print("\n📥 Download the models from /kaggle/working/models/ when complete!")

if __name__ == '__main__':
    main()