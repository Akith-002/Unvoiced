const video = document.getElementById("video");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const backendUrlInput = document.getElementById("backendUrl");
const sessionIdInput = document.getElementById("sessionId");
const intervalInput = document.getElementById("intervalMs");
const lastTime = document.getElementById("lastTime");
const predictionsDiv = document.getElementById("predictions");
const assembledDiv = document.getElementById("assembled");
const kidLetter = document.getElementById("kidLetter");
const useForm = document.getElementById("useForm");
const speakPred = document.getElementById("speakPred");

const canvas = document.getElementById("captureCanvas");
const ctx = canvas.getContext("2d");

let stream = null;
let timer = null;
let lastSpoken = "";
// game elements
const gameMode = document.getElementById("gameMode");
const gamePanel = document.getElementById("gamePanel");
const targetLetterEl = document.getElementById("targetLetter");
const scoreEl = document.getElementById("score");
const newTargetBtn = document.getElementById("newTargetBtn");
const timeLeftEl = document.getElementById("timeLeft");
const confidenceRing = document.getElementById("confidenceRing");
let game = { enabled: false, target: null, score: 0, timer: null, timeLeft: 8 };

// emoji map for letters (simple friendly mapping)
const emojiMap = {
  A: "🦅",
  B: "🐻",
  C: "🐱",
  D: "🐶",
  E: "🦄",
  F: "🍟",
  G: "🦒",
  H: "🐹",
  I: "🍦",
  J: "🕹️",
  K: "🦘",
  L: "🦁",
  M: "🐵",
  N: "🐧",
  O: "🐙",
  P: "🐼",
  Q: "👑",
  R: "🐰",
  S: "🐍",
  T: "🐯",
  U: "☂️",
  V: "🎻",
  W: "🦫",
  X: "❌",
  Y: "🪁",
  Z: "🦓",
  " ": "⬜",
  "?": "❓",
};

// simple success sound using WebAudio
let audioCtx = null;
function playBeep(freq = 880, duration = 0.12, type = "sine") {
  try {
    if (!audioCtx)
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    o.type = type;
    o.frequency.value = freq;
    g.gain.value = 0.12;
    o.connect(g);
    g.connect(audioCtx.destination);
    o.start();
    setTimeout(() => {
      o.stop();
    }, duration * 1000);
  } catch (e) {
    /* ignore */
  }
}

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

  // Update kid view big letter, speak if enabled, confidence ring and game
  try {
    updateKidView(json);
  } catch (e) {
    /* ignore */
  }
  try {
    updateConfidence(json);
  } catch (e) {
    /* ignore */
  }
  try {
    updateGame(json);
  } catch (e) {
    /* ignore */
  }
}

function updateKidView(json) {
  const top = (json.predictions && json.predictions[0]) || null;
  const label = top ? top.label : json.predicted_label || "?";
  if (label) {
    const L = String(label).toUpperCase();
    const emoji = emojiMap[L] || L;
    // if emoji, add class for smaller type
    if (emoji !== L) {
      kidLetter.classList.add("emoji");
      kidLetter.textContent = emoji;
    } else {
      kidLetter.classList.remove("emoji");
      kidLetter.textContent = L;
    }
    if (speakPred && speakPred.checked && lastSpoken !== label) {
      speakText(label);
      lastSpoken = label;
    }
  } else {
    kidLetter.textContent = "?";
  }
}

function updateConfidence(json) {
  // take top score or 0
  const top = (json.predictions && json.predictions[0]) || null;
  const score = top ? top.score || 0 : 0;
  // add classes based on thresholds
  if (!confidenceRing) return;
  confidenceRing.classList.remove("high", "low");
  if (score > 0.7) confidenceRing.classList.add("high");
  else if (score < 0.4) confidenceRing.classList.add("low");
  // scale ring a bit by confidence
  const scale = 0.95 + Math.max(0, score) * 0.15;
  confidenceRing.style.transform = `scale(${scale})`;
}

function updateGame(json) {
  if (!game.enabled) return;
  // if we have top prediction, compare to target
  const top = (json.predictions && json.predictions[0]) || null;
  if (!top) return;
  const predicted = String(top.label).toUpperCase();
  if (game.target && predicted === game.target) {
    // award points based on confidence
    const pts = Math.round((top.score || 0) * 10) + 1;
    game.score += pts;
    scoreEl.textContent = String(game.score);
    // visual & sound feedback
    kidLetter.classList.add("success-burst");
    setTimeout(() => kidLetter.classList.remove("success-burst"), 700);
    playBeep(1200, 0.08, "triangle");
    // auto-pick new target
    setTimeout(() => newTarget(), 700);
  }
}

function newTarget() {
  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const idx = Math.floor(Math.random() * letters.length);
  game.target = letters[idx];
  targetLetterEl.textContent = game.target;
  // reset timer
  game.timeLeft = 8;
  timeLeftEl.textContent = String(game.timeLeft);
  if (game.timer) clearInterval(game.timer);
  game.timer = setInterval(() => {
    game.timeLeft -= 1;
    timeLeftEl.textContent = String(game.timeLeft);
    if (game.timeLeft <= 0) {
      clearInterval(game.timer);
      game.timer = null;
      // penalty or new target
      game.timeLeft = 0;
      timeLeftEl.textContent = "0";
      playBeep(220, 0.12, "sine");
      newTarget();
    }
  }, 1000);
}

function speakText(txt) {
  try {
    if (!("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(txt);
    u.lang = "en-US";
    u.rate = 0.9;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  } catch (e) {
    // ignore speech errors
  }
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

// Game UI wiring
gameMode.addEventListener("change", () => {
  game.enabled = !!gameMode.checked;
  gamePanel.style.display = game.enabled ? "block" : "none";
  if (game.enabled) {
    game.score = 0;
    scoreEl.textContent = "0";
    newTarget();
  } else {
    if (game.timer) {
      clearInterval(game.timer);
      game.timer = null;
    }
    targetLetterEl.textContent = "-";
    timeLeftEl.textContent = "-";
  }
});
newTargetBtn.addEventListener("click", () => newTarget());

// Stop camera when page hidden
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopCamera();
});
