#!/usr/bin/env python3
"""
Simple Keras to TensorFlow Lite converter with float32 conversion
This version loads the model and converts it to float32 first to avoid mixed precision issues
"""

import tensorflow as tf
import numpy as np
import os
import sys

def convert_to_float32_and_tflite(keras_model_path, output_path):
    """
    Load Keras model, convert to float32, then to TensorFlow Lite
    """
    
    print(f"🔄 Loading Keras model from: {keras_model_path}")
    
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(keras_model_path)
        print("✅ Keras model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Create a new model with float32 precision
        print("🔄 Converting model to float32 precision...")
        
        # Get model input shape (without batch dimension)
        input_shape = model.input_shape[1:]  # Remove batch dimension
        
        # Create new input layer with float32
        inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)
        
        # Apply the original model but cast to float32
        x = tf.cast(inputs, tf.float32)
        outputs = model(x)
        outputs = tf.cast(outputs, tf.float32)
        
        # Create new model
        float32_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        print("✅ Model converted to float32")
        
    except Exception as e:
        print(f"❌ Error loading/converting Keras model: {e}")
        return False
    
    # Convert to TensorFlow Lite
    print("🔄 Converting to TensorFlow Lite...")
    
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(float32_model)
        
        # Use TensorFlow Select ops for compatibility
        converter.allow_custom_ops = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Basic optimization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        print("   Converting with TensorFlow Select ops...")
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"✅ TensorFlow Lite model saved to: {output_path}")
        
        # Get model size info
        original_size = os.path.getsize(keras_model_path) / (1024 * 1024)  # MB
        tflite_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        print(f"📊 Model Conversion Results:")
        print(f"   Original Keras model: {original_size:.2f} MB")
        print(f"   TensorFlow Lite model: {tflite_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting to TensorFlow Lite: {e}")
        print("\n💡 Trying simpler conversion without optimizations...")
        
        # Try simpler conversion
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(float32_model)
            converter.allow_custom_ops = True
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            # No optimizations this time
            
            tflite_model = converter.convert()
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
                
            print(f"✅ TensorFlow Lite model saved (simple conversion): {output_path}")
            return True
            
        except Exception as e2:
            print(f"❌ Simple conversion also failed: {e2}")
            return False

if __name__ == "__main__":
    models_dir = "models"
    keras_model_path = os.path.join(models_dir, "model.keras")
    tflite_output_path = os.path.join(models_dir, "model_simple.tflite")
    
    print("🚀 Simple Keras to TensorFlow Lite Converter")
    print("=" * 50)
    
    if not os.path.exists(keras_model_path):
        print(f"❌ Keras model not found at: {keras_model_path}")
        sys.exit(1)
    
    success = convert_to_float32_and_tflite(keras_model_path, tflite_output_path)
    
    if success:
        print("\n🎉 Simple conversion completed successfully!")
        print(f"   You can use {tflite_output_path} in your Flutter app")
        print("\n📝 Note: This model uses TensorFlow Select ops")
        print("   Make sure to include tf-select-ops in your Flutter app")
    else:
        print("\n❌ All conversion attempts failed!")
        print("💡 You may need to use the Keras model directly in your backend")