#!/usr/bin/env python3
"""
Convert Keras model to TensorFlow Lite format for mobile deployment
"""

import tensorflow as tf
import numpy as np
import os
import sys

def convert_keras_to_tflite(keras_model_path, output_path):
    """
    Convert Keras model to TensorFlow Lite format
    
    Args:
        keras_model_path: Path to the .keras model file
        output_path: Path where to save the .tflite model
    """
    
    print(f"🔄 Loading Keras model from: {keras_model_path}")
    
    # Load the Keras model
    try:
        model = tf.keras.models.load_model(keras_model_path)
        print("✅ Keras model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Error loading Keras model: {e}")
        return False
    
    # Convert to TensorFlow Lite
    print("🔄 Converting to TensorFlow Lite...")
    
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable TensorFlow Select operators for complex operations
        converter.allow_custom_ops = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite builtin ops
            tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow ops (fallback)
        ]
        
        # Optimization settings for mobile deployment
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set representative dataset for better quantization (optional)
        def representative_dataset():
            for _ in range(100):
                # Generate random data that matches your input shape (float32, not float16)
                data = np.random.random((1, 200, 200, 3)).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
        
        # Convert the model
        print("   Using TensorFlow Select ops for complex operations...")
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"✅ TensorFlow Lite model saved to: {output_path}")
        
        # Get model size info
        original_size = os.path.getsize(keras_model_path) / (1024 * 1024)  # MB
        tflite_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        compression_ratio = original_size / tflite_size
        
        print(f"📊 Model Conversion Results:")
        print(f"   Original Keras model: {original_size:.2f} MB")
        print(f"   TensorFlow Lite model: {tflite_size:.2f} MB")
        print(f"   Compression ratio: {compression_ratio:.2f}x smaller")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting to TensorFlow Lite: {e}")
        return False

def test_tflite_model(tflite_path, labels_path):
    """
    Test the TensorFlow Lite model with a random input
    """
    print(f"\n🧪 Testing TensorFlow Lite model...")
    
    try:
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Input type: {input_details[0]['dtype']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        print(f"   Output type: {output_details[0]['dtype']}")
        
        # Test with random data
        input_shape = input_details[0]['shape']
        input_data = np.random.random(input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get the output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data[0])
        confidence = np.max(output_data[0])
        
        # Load labels if available
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            predicted_label = labels[predicted_class] if predicted_class < len(labels) else f"Class_{predicted_class}"
        else:
            predicted_label = f"Class_{predicted_class}"
        
        print(f"✅ TensorFlow Lite model test successful!")
        print(f"   Predicted class: {predicted_class} ({predicted_label})")
        print(f"   Confidence: {confidence:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing TensorFlow Lite model: {e}")
        return False

if __name__ == "__main__":
    # Paths
    models_dir = "models"
    keras_model_path = os.path.join(models_dir, "model.keras")
    tflite_output_path = os.path.join(models_dir, "model.tflite")
    labels_path = os.path.join(models_dir, "training_set_labels.txt")
    
    print("🚀 Keras to TensorFlow Lite Converter")
    print("=" * 50)
    
    # Check if Keras model exists
    if not os.path.exists(keras_model_path):
        print(f"❌ Keras model not found at: {keras_model_path}")
        print("   Make sure you have downloaded model.keras from Kaggle")
        sys.exit(1)
    
    # Convert the model
    success = convert_keras_to_tflite(keras_model_path, tflite_output_path)
    
    if success:
        # Test the converted model
        test_tflite_model(tflite_output_path, labels_path)
        print("\n🎉 Conversion completed successfully!")
        print(f"   You can now use {tflite_output_path} in your Flutter app")
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)