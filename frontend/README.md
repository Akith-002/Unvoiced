Unvoiced — Webcam frontend

Simple static frontend to test the backend `/predict-image` endpoint using your desktop camera.

Quick start (local):

- Run the backend (in mock mode if you don't have TensorFlow installed):

  PowerShell:

  $env:USE_MOCK='1'; python backend\app.py

- Serve the frontend folder. From repository root you can run (PowerShell):

  cd frontend; python -m http.server 8000

  Then open http://localhost:8000 in your browser. Allow camera access when prompted.

Notes:

- The frontend captures a 200x200 center-crop from your webcam and sends it to the backend as either multipart/form-data (recommended) or base64 JSON.
- Provide the backend URL (default http://localhost:5000/predict-image) and an optional session ID to accumulate text.
- For quick testing without the real model, set environment variable USE_MOCK=1 which returns deterministic fake predictions.
