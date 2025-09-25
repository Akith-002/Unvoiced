const video = document.getElementById("video");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const backendUrlInput = document.getElementById("backendUrl");
const sessionIdInput = document.getElementById("sessionId");
const intervalInput = document.getElementById("intervalMs");
const lastTime = document.getElementById("lastTime");
const predictionsDiv = document.getElementById("predictions");
const assembledDiv = document.getElementById("assembled");
const useForm = document.getElementById("useForm");

const canvas = document.getElementById("captureCanvas");
const ctx = canvas.getContext("2d");

let stream = null;
let timer = null;

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    startBtn.disabled = true;
    stopBtn.disabled = false;
  } catch (e) {
    alert("Error accessing camera: " + e.message);
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.pause();
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

function captureFrame() {
  // Draw centered crop to 200x200
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const size = Math.min(vw, vh);
  const sx = (vw - size) / 2;
  const sy = (vh - size) / 2;
  ctx.drawImage(video, sx, sy, size, size, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
}

async function sendFrame() {
  if (!stream) return;
  const blob = await captureFrame();
  const backend = backendUrlInput.value.trim();
  if (!backend) {
    alert("Set backend URL");
    return;
  }

  const session_id = sessionIdInput.value.trim() || undefined;
  const useMultipart = useForm.checked;

  lastTime.textContent = "Sending...";

  try {
    let resp;
    if (useMultipart) {
      const fd = new FormData();
      fd.append("image", blob, "frame.jpg");
      if (session_id) fd.append("session_id", session_id);
      resp = await fetch(backend, { method: "POST", body: fd });
    } else {
      // send base64 JSON
      const arr = await blob.arrayBuffer();
      const b64 = arrayBufferToBase64(arr);
      const payload = { image_base64: b64 };
      if (session_id) payload.session_id = session_id;
      resp = await fetch(backend, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    }

    const json = await resp.json();
    if (!resp.ok) {
      predictionsDiv.textContent = "Error: " + (json.error || resp.statusText);
      lastTime.textContent = "Error";
      return;
    }
    renderPredictions(json);
    lastTime.textContent = new Date().toLocaleTimeString();
  } catch (e) {
    lastTime.textContent = "Req failed";
    predictionsDiv.textContent = "Request failed: " + e.message;
  }
}

function arrayBufferToBase64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function renderPredictions(json) {
  const preds = json.predictions || [];
  predictionsDiv.innerHTML = "";
  const ul = document.createElement("div");
  ul.style.fontSize = "14px";
  preds.slice(0, 5).forEach((p) => {
    const el = document.createElement("div");
    el.textContent = `${p.label} — ${Math.round(p.score * 10000) / 100}%`;
    ul.appendChild(el);
  });
  predictionsDiv.appendChild(ul);
  assembledDiv.textContent =
    (json.assembled_text || "") +
    (json.predicted_label ? "  (top: " + json.predicted_label + ")" : "");
}

startBtn.addEventListener("click", async () => {
  await startCamera();
  const ms = Math.max(100, parseInt(intervalInput.value || "600", 10));
  timer = setInterval(sendFrame, ms);
  // send first frame immediately
  sendFrame();
});

stopBtn.addEventListener("click", () => {
  stopCamera();
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
});

// allow pressing Enter in backend URL to test once
backendUrlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendFrame();
});

// Handy: allow clicking video to capture one frame
video.addEventListener("click", () => {
  sendFrame();
});

// Stop camera when page hidden
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopCamera();
});
