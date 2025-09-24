import os
import io
import json

# Use real model
os.environ['USE_MOCK_MODEL'] = '0'
from app import app

def test_session_text_accumulation():
    """Test that session-based text accumulation works with real model"""
    app.testing = True
    client = app.test_client()
    
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_images_dir = os.path.join(repo_root, 'Test Images')
    
    session_id = 'demo-session'
    
    # Send A, then B, then C
    test_sequence = ['A_test.jpg', 'B_test.jpg', 'C_test.jpg']
    
    print("Testing session text accumulation:")
    print(f"Session ID: {session_id}")
    print("-" * 40)
    
    for test_file in test_sequence:
        test_path = os.path.join(test_images_dir, test_file)
        if not os.path.exists(test_path):
            continue
            
        with open(test_path, 'rb') as f:
            img_bytes = f.read()
        
        data = {
            'session_id': session_id,
            'image': (io.BytesIO(img_bytes), test_file)
        }
        
        resp = client.post('/predict-image', data=data, content_type='multipart/form-data')
        
        if resp.status_code == 200:
            result = resp.get_json()
            print(f'{test_file}: predicted="{result["predicted_label"]}", assembled_text="{result["assembled_text"]}"')
    
    print("\nNow testing SPACE gesture:")
    # Test with space gesture if available
    space_test = 'space_test.jpg'
    space_path = os.path.join(test_images_dir, space_test)
    if os.path.exists(space_path):
        with open(space_path, 'rb') as f:
            img_bytes = f.read()
        
        data = {
            'session_id': session_id,
            'image': (io.BytesIO(img_bytes), space_test)
        }
        
        resp = client.post('/predict-image', data=data, content_type='multipart/form-data')
        
        if resp.status_code == 200:
            result = resp.get_json()
            print(f'{space_test}: predicted="{result["predicted_label"]}", assembled_text="{result["assembled_text"]}"')

if __name__ == '__main__':
    test_session_text_accumulation()