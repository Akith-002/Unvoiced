"""
Complete training pipeline for ASL sign language classification using MobileNetV3.
Optimized for 6GB GPU with memory management and mobile deployment.

Usage:
    python train_finetune.py --dataset_dir dataset --epochs 50 --batch_size 32
    
Expected directory structure:
    dataset/
        train/
            A/
                *.jpg
            B/
                *.jpg
            ...
            SPACE/
            DELETE/  
            NOTHING/
        val/
            A/
            B/
            ...
        test/
            A/
            B/
            ...
"""

import os
import argparse
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
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure TensorFlow for optimal performance
def configure_gpu():
    """Configure GPU/CPU for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision for better GPU utilization
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info(f"GPU configured with memory growth. Mixed precision enabled.")
            logger.info(f"Available GPUs: {len(gpus)}")
            
        except RuntimeError as e:
            logger.error(f"GPU configuration failed: {e}")
    else:
        logger.info("No GPU detected, optimizing for CPU training...")
        # Optimize CPU performance
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
        # Use float32 for CPU (mixed precision can cause issues)
        tf.keras.mixed_precision.set_global_policy('float32')
        logger.info("CPU optimization enabled - using all available cores")

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
    
    # Get class names for later use
    class_names = dataset.class_names
    
    return dataset, class_names

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
    """Create MobileNetV3Large model with custom head"""
    
    # Load pre-trained MobileNetV3Large
    base_model = MobileNetV3Large(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', dtype='float32')  # Ensure float32 output for mixed precision
    ])
    
    return model, base_model

def calculate_class_weights(dataset, class_names):
    """Calculate class weights for imbalanced dataset"""
    
    # Count samples per class
    class_counts = {}
    total_samples = 0
    
    for images, labels in dataset:
        labels_np = labels.numpy()
        for label_vec in labels_np:
            class_idx = np.argmax(label_vec)
            class_name = class_names[class_idx]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_samples += 1
    
    # Calculate weights (inverse frequency)
    class_weights = {}
    for idx, class_name in enumerate(class_names):
        count = class_counts.get(class_name, 1)
        weight = total_samples / (len(class_names) * count)
        class_weights[idx] = weight
    
    logger.info(f"Class distribution: {class_counts}")
    logger.info(f"Class weights: {class_weights}")
    
    return class_weights

def train_model(model, base_model, train_ds, val_ds, class_weights, epochs=50, output_dir='output'):
    """Two-phase training: frozen base + fine-tuning"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 1: Train only the top layers
    logger.info("=== Phase 1: Training top layers (base frozen) ===")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for phase 1
    callbacks_phase1 = [
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model_phase1.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train phase 1
    phase1_epochs = min(15, epochs // 3)
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=callbacks_phase1,
        class_weight=class_weights,
        verbose=1
    )
    
    # Phase 2: Fine-tune some layers of the base model
    logger.info("=== Phase 2: Fine-tuning base model ===")
    
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = len(base_model.layers) - 30  # Unfreeze last 30 layers
    
    # Freeze all layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    logger.info(f"Fine-tuning from layer {fine_tune_at} onwards")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for phase 2
    callbacks_phase2 = [
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model_final.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train phase 2
    phase2_epochs = epochs - phase1_epochs
    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase2_epochs,
        callbacks=callbacks_phase2,
        class_weight=class_weights,
        verbose=1
    )
    
    # Combine histories
    history = {
        'phase1': history_phase1.history,
        'phase2': history_phase2.history
    }
    
    return model, history

def evaluate_model(model, test_ds, class_names, output_dir):
    """Evaluate model and generate detailed report"""
    
    logger.info("=== Evaluating model on test set ===")
    
    # Get predictions and true labels
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_pred_proba.extend(predictions)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate metrics
    test_accuracy = np.mean(y_true == y_pred)
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Save detailed report
    report_data = {
        'test_accuracy': float(test_accuracy),
        'per_class_metrics': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    with open(os.path.join(output_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cmap='Blues'
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Per-class accuracy
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
            logger.info(f"{class_name}: {class_acc:.4f}")
    
    return report_data

def export_models(model, class_names, output_dir):
    """Export model in multiple formats"""
    
    logger.info("=== Exporting models ===")
    
    # 1. Save as SavedModel (recommended for serving)
    savedmodel_path = os.path.join(output_dir, 'saved_model')
    model.save(savedmodel_path)
    logger.info(f"SavedModel exported to: {savedmodel_path}")
    
    # 2. Save class names
    labels_path = os.path.join(output_dir, 'training_set_labels.txt')
    with open(labels_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name.lower()}\n")
    logger.info(f"Labels saved to: {labels_path}")
    
    # 3. Convert to TensorFlow Lite for mobile
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
        
        # Optimize for mobile
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = None  # You can add a representative dataset here
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        # Convert to TFLite
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = os.path.join(output_dir, 'model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TFLite model exported to: {tflite_path}")
        logger.info(f"TFLite model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        logger.error(f"TFLite conversion failed: {e}")
    
    # 4. Export frozen graph (for compatibility with existing code)
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        
        # Convert to concrete function
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
        )
        
        # Convert to frozen graph
        frozen_func = convert_variables_to_constants_v2(full_model)
        
        # Save frozen graph
        tf.io.write_graph(
            graph_or_graph_def=frozen_func.graph,
            logdir=output_dir,
            name='trained_model_graph.pb',
            as_text=False
        )
        
        logger.info(f"Frozen graph exported to: {os.path.join(output_dir, 'trained_model_graph.pb')}")
        
    except Exception as e:
        logger.error(f"Frozen graph export failed: {e}")

def plot_training_history(history, output_dir):
    """Plot training curves"""
    
    # Combine phase 1 and phase 2 histories
    metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
    combined_history = {}
    
    for metric in metrics:
        combined_history[metric] = []
        if metric in history['phase1']:
            combined_history[metric].extend(history['phase1'][metric])
        if metric in history['phase2']:
            combined_history[metric].extend(history['phase2'][metric])
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Loss
    axes[0, 0].plot(combined_history['loss'], label='Training Loss')
    axes[0, 0].plot(combined_history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(combined_history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(combined_history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate (if available)
    axes[1, 0].text(0.5, 0.5, 'Phase 1: Frozen base\nPhase 2: Fine-tuning', 
                   ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Training Phases')
    
    # Summary stats
    best_val_acc = max(combined_history['val_accuracy'])
    best_val_loss = min(combined_history['val_loss'])
    final_train_acc = combined_history['accuracy'][-1]
    
    summary_text = f"""
    Final Training Accuracy: {final_train_acc:.4f}
    Best Validation Accuracy: {best_val_acc:.4f}
    Best Validation Loss: {best_val_loss:.4f}
    Total Epochs: {len(combined_history['loss'])}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, 
                   ha='left', va='center', transform=axes[1, 1].transAxes,
                   fontsize=12)
    axes[1, 1].set_title('Training Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save history as JSON
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(combined_history, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Train ASL classification model')
    parser.add_argument('--dataset_dir', type=str, default='dataset', 
                       help='Directory containing train/val/test subdirectories')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save trained models and results')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Total number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (16 for CPU, 32+ for GPU)')
    parser.add_argument('--img_size', type=int, nargs=2, default=[200, 200],
                       help='Image size (height width)')
    
    args = parser.parse_args()
    
    # Configure GPU
    configure_gpu()
    
    # Verify dataset structure
    dataset_dir = Path(args.dataset_dir)
    required_dirs = ['train', 'val', 'test']
    
    for split_dir in required_dirs:
        split_path = dataset_dir / split_dir
        if not split_path.exists():
            logger.error(f"Required directory not found: {split_path}")
            logger.error("Please organize your dataset as: dataset/train/CLASS/*.jpg, dataset/val/CLASS/*.jpg, dataset/test/CLASS/*.jpg")
            return
    
    logger.info(f"Dataset directory: {args.dataset_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Image size: {args.img_size}")
    logger.info(f"Epochs: {args.epochs}")
    
    # Create datasets
    logger.info("Loading datasets...")
    
    train_ds, class_names = create_dataset(
        dataset_dir / 'train',
        batch_size=args.batch_size,
        img_size=tuple(args.img_size)
    )
    
    val_ds, _ = create_dataset(
        dataset_dir / 'val',
        batch_size=args.batch_size,
        img_size=tuple(args.img_size)
    )
    
    test_ds, _ = create_dataset(
        dataset_dir / 'test',
        batch_size=args.batch_size,
        img_size=tuple(args.img_size)
    )
    
    logger.info(f"Found {len(class_names)} classes: {class_names}")
    
    # Apply augmentation
    train_ds = augment_dataset(train_ds, is_training=True)
    val_ds = augment_dataset(val_ds, is_training=False)
    test_ds = augment_dataset(test_ds, is_training=False)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_ds, class_names)
    
    # Create model
    logger.info("Creating model...")
    model, base_model = create_model(len(class_names), tuple(args.img_size))
    
    logger.info(f"Model created. Total parameters: {model.count_params():,}")
    
    # Train model
    model, history = train_model(
        model, base_model, train_ds, val_ds, class_weights, 
        epochs=args.epochs, output_dir=args.output_dir
    )
    
    # Evaluate model
    evaluation_results = evaluate_model(model, test_ds, class_names, args.output_dir)
    
    # Plot training history
    plot_training_history(history, args.output_dir)
    
    # Export models
    export_models(model, class_names, args.output_dir)
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved in: {args.output_dir}")
    
    # Print final summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Test Accuracy: {evaluation_results['test_accuracy']:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print(f"TFLite model: {args.output_dir}/model.tflite")
    print(f"Labels file: {args.output_dir}/training_set_labels.txt")
    print("="*50)

if __name__ == '__main__':
    main()