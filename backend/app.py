import os
import io
import base64
import json
import threading
from PIL import Image
try:
    # Some environments may have an older or patched Pillow where
    # ImageFile.StubHandler (and StubImageFile) is missing. A few
    # third-party libraries expect these attributes to exist. Provide a
    # lightweight compatibility shim so image decoding doesn't fail with
    # "module 'PIL.ImageFile' has no attribute 'StubHandler'".
    import PIL.ImageFile as _ImageFile
    if not hasattr(_ImageFile, 'StubHandler'):
        class StubHandler:
            """Minimal stub for compatibility; real behavior isn't needed
            for our use-case (we open/convert images via PIL.Image).
            """
            pass

        class StubImageFile(object):
            pass

        _ImageFile.StubHandler = StubHandler
        _ImageFile.StubImageFile = StubImageFile
except Exception:
    # If the shim fails for any reason, continue; we'll let actual
    # image operations surface errors later and return them to the client.
    pass
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import random

# Model paths - using the new trained models
BACKEND_ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BACKEND_ROOT, 'models')
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, 'model.keras')
LABELS_PATH = os.path.join(MODEL_DIR, 'training_set_labels.txt')

# Fallback to old paths if new models don't exist
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OLD_GRAPH_PATH = os.path.join(REPO_ROOT, 'trained_model_graph.pb')
OLD_LABELS_PATH = os.path.join(REPO_ROOT, 'training_set_labels.txt')

app = Flask(__name__)
CORS(app)

# Support a mock mode to allow smoke tests without TensorFlow installed.
USE_MOCK = os.environ.get('USE_MOCK_MODEL', '0') == '1'

if USE_MOCK:
    # Minimal fake labels and a deterministic fake predictor
    label_lines = ['a', 'b', 'c', 'space', 'nothing', 'del']
    sess = None

    def predict_image_bytes(image_bytes, top_k=3):
        # deterministic fake: return 'a' with high score, 'space' second
        results = [
            {'label': 'a', 'score': 0.95},
            {'label': 'space', 'score': 0.03},
            {'label': 'nothing', 'score': 0.02},
        ]
        return results[:top_k]

else:
    # Try to use the new Keras model first, fallback to old frozen graph
    import tensorflow as tf
    import numpy as np
    
    # Check which model to use
    use_keras_model = os.path.exists(KERAS_MODEL_PATH)
    
    if use_keras_model:
        print(f"Loading new Keras model from: {KERAS_MODEL_PATH}")
        # Load the new Keras model
        model = tf.keras.models.load_model(KERAS_MODEL_PATH)
        
        # Load labels
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError('labels file not found: ' + LABELS_PATH)
        with open(LABELS_PATH, 'r') as f:
            label_lines = [line.strip() for line in f.readlines()]
        
        sess = None  # Not needed for Keras model
        
        def predict_image_bytes(image_bytes, top_k=3):
            """Predict using the new Keras model"""
            # Open and preprocess the image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to 200x200 (model input size)
            image = image.resize((200, 200))
            
            # Convert to numpy array and normalize
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Make prediction
            predictions = model.predict(img_array, verbose=0)[0]
            
            # Get top K predictions
            top_indices = np.argsort(predictions)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(label_lines):
                    results.append({
                        'label': label_lines[idx],
                        'score': float(predictions[idx])
                    })
            
            return results
        
    else:
        print(f"Keras model not found, using fallback frozen graph from: {OLD_GRAPH_PATH}")
        # Use TF1 compatibility mode to load the existing frozen graph
        tf = tf.compat.v1
        tf.disable_v2_behavior()

        # Load labels - try new location first, then old
        labels_file = LABELS_PATH if os.path.exists(LABELS_PATH) else OLD_LABELS_PATH
        if not os.path.exists(labels_file):
            raise FileNotFoundError('labels file not found: ' + labels_file)
        with tf.gfile.GFile(labels_file) as f:
            label_lines = [line.rstrip() for line in f]

        # Load graph
        if not os.path.exists(OLD_GRAPH_PATH):
            raise FileNotFoundError('graph file not found: ' + OLD_GRAPH_PATH)
        with tf.gfile.FastGFile(OLD_GRAPH_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        # Create a single session for the app
        sess = tf.Session()
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        def predict_image_bytes(image_bytes, top_k=3):
            """Run the model on JPEG image bytes and return top predictions.

            The original project expects 'DecodeJpeg/contents:0' to accept raw
            JPEG bytes, so we forward the bytes directly.
            """
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_bytes})
            probs = predictions[0]
            top_idx = probs.argsort()[-top_k:][::-1]
            results = []
            for i in top_idx:
                results.append({'label': label_lines[i], 'score': float(probs[i])})
            return results


# Simple in-memory session store to assemble text across multiple frames
# Keyed by session_id (string). This is ephemeral (process memory).
session_store = {}
session_lock = threading.Lock()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict-image', methods=['POST'])
def predict_image():
    # Accept multipart/form-data file or JSON with base64 string
    image_bytes = None

    # Log incoming form/json keys (do not log large or binary values)
    try:
        form_keys = list(request.form.keys()) if request.form else []
        json_keys = []
        try:
            j = request.get_json(silent=True)
            if isinstance(j, dict):
                json_keys = list(j.keys())
        except Exception:
            json_keys = []

        print(f"[DEBUG] incoming request keys - form: {form_keys}, json: {json_keys}")
        # If assembled_text is being sent from frontend it will appear in one of these
        if 'assembled_text' in form_keys or 'assembled_text' in json_keys:
            print('[DEBUG] incoming request contains assembled_text field')
    except Exception:
        pass

    # 1) multipart file
    if 'image' in request.files:
        f = request.files['image']
        image_bytes = f.read()

    # 2) raw base64 in JSON body
    elif request.is_json:
        data = request.get_json()
        b64 = data.get('image_base64') or data.get('image')
        if b64:
            # strip data URL prefix if present
            if b64.startswith('data:') and ';base64,' in b64:
                b64 = b64.split(';base64,', 1)[1]
            image_bytes = base64.b64decode(b64)

    # 3) raw body bytes
    else:
        raw = request.get_data()
        if raw:
            image_bytes = raw

    if not image_bytes:
        return jsonify({'error': 'no image provided'}), 400

    # Debug logging for image format analysis
    print(f"[DEBUG] Received image: {len(image_bytes)} bytes")
    print(f"[DEBUG] First 10 bytes: {image_bytes[:10].hex() if len(image_bytes) >= 10 else 'N/A'}")
    print(f"[DEBUG] Is JPEG (starts with FFD8): {len(image_bytes) >= 2 and image_bytes[0] == 0xFF and image_bytes[1] == 0xD8}")
    
    # If the bytes are not JPEG, try to convert using Pillow to JPEG
    try:
        # Quick check: JPEG files start with 0xFF 0xD8
        if not (len(image_bytes) >= 2 and image_bytes[0] == 0xFF and image_bytes[1] == 0xD8):
            print(f"[DEBUG] Converting non-JPEG to JPEG")
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            print(f"[DEBUG] Original image: {img.size}, {img.mode}")
            out = io.BytesIO()
            img.save(out, format='JPEG', quality=90)
            image_bytes = out.getvalue()
            print(f"[DEBUG] Converted image: {len(image_bytes)} bytes")
    except Exception as e:
        # If conversion fails, continue and let TF attempt decode
        print(f"[DEBUG] Image conversion failed: {e}")
        pass

    try:
        results = predict_image_bytes(image_bytes, top_k=5)
    except Exception as e:
        return jsonify({'error': 'model inference failed', 'details': str(e)}), 500

    top = results[0] if results else None

    # Build a simple textual output from top label (preserve case from labels file)
    top_label = top['label'] if top else None

    # Optional session_id to accumulate letters into a current word
    session_id = None
    current_text = None
    if request.is_json:
        session_id = request.get_json().get('session_id')
    else:
        session_id = request.form.get('session_id') or request.args.get('session_id')

    if session_id:
        with session_lock:
            state = session_store.get(session_id, {'current_word': ''})
            cw = state.get('current_word', '')

            if top_label:
                token = str(top_label).strip().lower()
                # Interpret special tokens from the label set
                if token == 'space':
                    # finalize current word (append a space)
                    if len(cw) > 0:
                        cw = cw + ' '
                elif token == 'del':
                    # delete last char
                    if len(cw) > 0:
                        cw = cw[:-1]
                elif token == 'nothing':
                    # ignore
                    pass
                else:
                    # normal letter: append (convert to upper for readability)
                    cw = cw + token.upper()

            state['current_word'] = cw
            session_store[session_id] = state
            current_text = cw

    resp = {
        'predictions': results,
        'top_prediction': top,
        'predicted_label': top_label,
        'assembled_text': current_text
    }

    # Debugging: save incoming images and prediction metadata occasionally.
    # Behavior:
    # - If DEBUG_SAVE_IMAGES=1 => always save
    # - Otherwise, save with probability SAVE_SAMPLE_RATE (0.05 by default)
    #   only when top_label is truthy (e.g. to reduce noise).
    # - Existing behavior of writing to backend/debug_received/ is preserved.
    try:
        env_force = os.environ.get('DEBUG_SAVE_IMAGES', '0') == '1'
        # Sampling rate: float between 0 and 1 (default 0.05 == 5%)
        try:
            sample_rate = float(os.environ.get('SAVE_SAMPLE_RATE', '0.05'))
            if sample_rate < 0.0 or sample_rate > 1.0:
                sample_rate = 0.05
        except Exception:
            sample_rate = 0.05

        should_sample = random.random() < sample_rate
        # Only save when forced, or when sampling succeeds and we have a label
        save_debug = env_force or (top_label is not None and should_sample)

        if save_debug:
            debug_dir = os.path.join(BACKEND_ROOT, 'debug_received')
            os.makedirs(debug_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            sess = session_id or 'no-session'
            # sanitize label for filename
            safe_label = str(top_label).replace(os.path.sep, '_') if top_label is not None else 'None'
            fname = f"{ts}_sess-{sess}_pred-{safe_label}.jpg"
            fpath = os.path.join(debug_dir, fname)
            # Write raw image bytes to disk for inspection
            try:
                with open(fpath, 'wb') as out:
                    out.write(image_bytes)
            except Exception:
                # fallback: try to convert via PIL then save
                try:
                    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    img.save(fpath, format='JPEG')
                except Exception:
                    # give up silently; we don't want to break the API
                    pass

            # Also print a concise log to stdout so the server logs show the event.
            # Truncate assembled_text in logs to avoid extremely long entries.
            try:
                if current_text is None:
                    asm_disp = 'None'
                else:
                    # Keep only first 120 chars and indicate truncation
                    asm_disp = (str(current_text)[:120] + '...') if len(str(current_text)) > 120 else str(current_text)
                print(f"[DEBUG] saved image -> {fpath}; predicted={top_label}; assembled={asm_disp}")
            except Exception:
                # Fallback to original minimal log if formatting fails
                print(f"[DEBUG] saved image -> {fpath}; predicted={top_label}; assembled=<error formatting>")
    except Exception as ex:
        # Do not allow debugging code to break normal responses
        print('[DEBUG] failed to save debug image:', ex)
    return jsonify(resp)


if __name__ == '__main__':
    # Allow configuring host/port via env for simple deployments
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '5000'))
    app.run(host=host, port=port, debug=False)
