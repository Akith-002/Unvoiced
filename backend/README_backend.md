# Unvoiced backend

This small Flask app exposes the existing TensorFlow frozen graph (`trained_model_graph.pb`) as a REST API so a mobile frontend (for example a Flutter app) can send images and receive predictions.

- Endpoints
- GET /health — simple health check
- POST /predict-image — predict the letter in an image

- multipart/form-data with a file field named `image`
- JSON with `image_base64` containing a base64-encoded image (data URL or raw base64)
- raw image bytes in the request body

Accepts

- multipart/form-data with a file field named `image`
- JSON with `image_base64` containing a base64-encoded image (data URL or raw base64)
- raw image bytes in the request body

## Session text assembly

If your Flutter app wants to build words from multiple frames (for example streaming camera frames), include a `session_id` string in the request (either as a form field or as JSON key `session_id`). The backend will keep an ephemeral in-memory string for that session and will append letters according to the model's top label. Special labels handled:

- `space` — appends a space (finalizes the current word)
- `del` — deletes the last character
- `nothing` — ignored

Example response now includes `assembled_text` which is the current string accumulated for that session.

- multipart/form-data with a file field named `image`
- JSON with `image_base64` containing a base64-encoded image (data URL or raw base64)
- raw image bytes in the request body

Response
{
"predictions": [ {"label": "a", "score": 0.95}, ... ],
"top_prediction": {"label": "a", "score": 0.95}
}

Notes for Flutter integration

1. Send camera-captured image as multipart/form-data using `http` or `dio` packages.
2. Alternatively, send base64 via JSON where `image_base64` is the base64 string.

Example (Flutter, using http package):

```dart
var request = http.MultipartRequest('POST', Uri.parse('https://your-backend/predict-image'));
request.fields['session_id'] = 'device123';
request.files.add(await http.MultipartFile.fromPath('image', imagePath));
var res = await request.send();
var body = await res.stream.bytesToString();
var json = jsonDecode(body);
// json['assembled_text'] contains the running text for session 'device123'
```

TTS recommendation

- The original repo uses server-side TTS (gTTS). For mobile apps, prefer doing text-to-speech on the client to reduce latency and avoid transferring audio files. Flutter has `flutter_tts` which works offline.

Deployment

- You can containerize with Docker, or deploy to a simple VM. Note the model uses TensorFlow 1 style frozen graph; the requirements pin TensorFlow 2 but uses compat.v1 to load the graph.
