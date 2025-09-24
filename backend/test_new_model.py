#!/usr/bin/env python3
"""
Test the new high-accuracy model with test images
"""

import requests
import json
import os
import time

def test_backend_with_image(image_path, backend_url="http://127.0.0.1:5000"):
    """Test the backend with a specific image"""
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    # Read the image file
    with open(image_path, 'rb') as f:
        files = {'image': f}
        
        try:
            response = requests.post(f"{backend_url}/predict-image", files=files, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                print(f"❌ Request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            return None

def test_multiple_images():
    """Test multiple images and show results"""
    
    # Test images directory
    test_dir = "../Test Images"
    
    # Some test images to try
    test_images = [
        "A_test.jpg",
        "B_test.jpg", 
        "C_test.jpg",
        "space_test.jpg",
        "nothing_test.jpg"
    ]
    
    print("🧪 Testing New High-Accuracy Model")
    print("=" * 50)
    
    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        
        expected_letter = img_name.split('_')[0].lower()
        print(f"\n📸 Testing: {img_name} (Expected: {expected_letter})")
        
        start_time = time.time()
        result = test_backend_with_image(img_path)
        end_time = time.time()
        
        if result:
            predictions = result.get('predictions', [])
            if predictions:
                top_prediction = predictions[0]
                confidence = top_prediction['score']
                predicted_label = top_prediction['label']
                
                # Check if prediction is correct
                is_correct = predicted_label.lower() == expected_letter
                status = "✅" if is_correct else "❌"
                
                print(f"   {status} Predicted: {predicted_label} (confidence: {confidence:.3f})")
                print(f"   ⏱️  Inference time: {(end_time - start_time)*1000:.1f}ms")
                
                # Show top 3 predictions
                print("   🔍 Top 3 predictions:")
                for i, pred in enumerate(predictions[:3]):
                    print(f"      {i+1}. {pred['label']}: {pred['score']:.3f}")
            else:
                print("   ❌ No predictions returned")
        else:
            print("   ❌ Failed to get prediction")
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    test_multiple_images()