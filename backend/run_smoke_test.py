import os
import io
import json

# Ensure mock mode is enabled before importing the app so the app uses the
# mocked predictor when TensorFlow is not installed.
os.environ['USE_MOCK_MODEL'] = '1'
from app import app


# Run the smoke test against the Flask app using the Flask test client.
def run():
    # Use the app in testing mode
    app.testing = True
    client = app.test_client()

    # Test image path from repo Test Images
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_img = os.path.join(repo_root, 'Test Images', 'A_test.jpg')
    if not os.path.exists(test_img):
        print('Test image not found:', test_img)
        return

    # Read file bytes and prepare an in-memory file object for multipart
    with open(test_img, 'rb') as f:
        img_bytes = f.read()

    data = {
        'session_id': 'smoke-session-1',
        'image': (io.BytesIO(img_bytes), 'A_test.jpg')
    }

    resp = client.post('/predict-image', data=data, content_type='multipart/form-data')

    print('Status code:', resp.status_code)
    try:
        print(json.dumps(resp.get_json(), indent=2))
    except Exception as e:
        print('Failed to parse JSON response:', e)


if __name__ == '__main__':
    run()
