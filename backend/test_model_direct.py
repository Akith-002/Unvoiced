#!/usr/bin/env python3
"""
Direct test of the new Keras model without Flask server
"""

import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

# Add the backend directory to the path
backend_dir = os.path.dirname(__file__)
sys.path.insert(0, backend_dir)

def test_model_directly():
    """Test the Keras model directly"""
    
    # Model paths
    model_path = os.path.join(backend_dir, 'models', 'model.keras')
    labels_path = os.path.join(backend_dir, 'models', 'training_set_labels.txt')
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
        
    if not os.path.exists(labels_path):
        print(f"❌ Labels not found: {labels_path}")
        return
    
    # Load model and labels
    print("🔄 Loading model and labels...")
    model = tf.keras.models.load_model(model_path)
    
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    print(f"✅ Model loaded successfully")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Number of classes: {len(labels)}")
    
    # Test with some images
    test_dir = "../Test Images"
    test_images = [
        ("A_test.jpg", "a"),
        ("B_test.jpg", "b"), 
        ("space_test.jpg", "space"),
        ("nothing_test.jpg", "nothing")
    ]
    
    print("\n🧪 Testing with sample images:")
    print("=" * 50)
    
    correct_predictions = 0
    total_predictions = 0
    
    for img_name, expected in test_images:
        img_path = os.path.join(test_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue
            
        # Load and preprocess image
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((200, 200))
        
        # Convert to array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top prediction
        top_idx = np.argmax(predictions)
        predicted_label = labels[top_idx]
        confidence = predictions[top_idx]
        
        # Check if correct
        is_correct = predicted_label.lower() == expected.lower()
        status = "✅" if is_correct else "❌"
        
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        print(f"{status} {img_name}: {predicted_label} ({confidence:.3f}) [Expected: {expected}]")
        
        # Show top 3 predictions
        top_3_indices = np.argsort(predictions)[::-1][:3]
        print(f"   Top 3: ", end="")
        for i, idx in enumerate(top_3_indices):
            print(f"{labels[idx]}({predictions[idx]:.3f})", end="")
            if i < 2:
                print(", ", end="")
        print()
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n📊 Test Results:")
        print(f"   Correct: {correct_predictions}/{total_predictions}")
        print(f"   Accuracy: {accuracy:.1%}")
        
        if accuracy >= 0.8:
            print("🎉 Great! Model is performing well!")
        elif accuracy >= 0.6:
            print("👍 Good performance!")
        else:
            print("⚠️  Model may need more training or different images")
    
    print("\n✅ Direct model test completed!")

if __name__ == "__main__":
    test_model_directly()