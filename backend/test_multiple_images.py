import os
import io
import json

# Ensure we're using the real model (not mock)
os.environ['USE_MOCK_MODEL'] = '0'
from app import app

def test_multiple_images():
    """Test with different images to verify real model inference"""
    app.testing = True
    client = app.test_client()
    
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_images_dir = os.path.join(repo_root, 'Test Images')
    
    # Test a few different letters
    test_files = ['A_test.jpg', 'B_test.jpg', 'C_test.jpg']
    
    for test_file in test_files:
        test_path = os.path.join(test_images_dir, test_file)
        if not os.path.exists(test_path):
            print(f'Skipping {test_file} - not found')
            continue
            
        with open(test_path, 'rb') as f:
            img_bytes = f.read()
        
        data = {
            'session_id': f'test-{test_file}',
            'image': (io.BytesIO(img_bytes), test_file)
        }
        
        resp = client.post('/predict-image', data=data, content_type='multipart/form-data')
        
        if resp.status_code == 200:
            result = resp.get_json()
            print(f'{test_file}: predicted_label="{result["predicted_label"]}", score={result["top_prediction"]["score"]:.3f}')
        else:
            print(f'{test_file}: ERROR - Status {resp.status_code}')

if __name__ == '__main__':
    test_multiple_images()