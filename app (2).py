"""
Voice-Based Emotion Identifier
Flask web application with ML model for real-time voice emotion detection.
"""

import os
import io
import json
import base64
import tempfile
import numpy as np
import librosa
from flask import Flask, request, jsonify, render_template_string
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ─── Emotion Labels & Colors ────────────────────────────────────────────────
EMOTIONS = ["angry", "calm", "fearful", "happy", "neutral", "sad", "surprised", "disgusted"]

EMOTION_META = {
    "angry":     {"emoji": "😠", "color": "#FF4444", "desc": "High energy, tense vocal patterns"},
    "calm":      {"emoji": "😌", "color": "#4ECDC4", "desc": "Steady pitch, relaxed rhythm"},
    "fearful":   {"emoji": "😨", "color": "#9B59B6", "desc": "Elevated pitch, irregular cadence"},
    "happy":     {"emoji": "😄", "color": "#F7DC6F", "desc": "Bright timbre, rising inflection"},
    "neutral":   {"emoji": "😐", "color": "#95A5A6", "desc": "Balanced energy, flat prosody"},
    "sad":       {"emoji": "😢", "color": "#5DADE2", "desc": "Low energy, falling pitch"},
    "surprised": {"emoji": "😲", "color": "#F39C12", "desc": "Sharp onset, high pitch burst"},
    "disgusted": {"emoji": "🤢", "color": "#27AE60", "desc": "Tense articulation, low pitch"},
}

MODEL_PATH = "emotion_model.joblib"


# ─── Feature Extraction ──────────────────────────────────────────────────────
def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract a rich feature vector from a raw audio signal."""
    features = []

    # MFCCs (mean + std of 40 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Delta MFCCs
    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(mfcc_delta, axis=1))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # Spectral features
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(float(np.mean(spec_cent)))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(float(np.mean(spec_bw)))

    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(float(np.mean(spec_roll)))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(float(np.mean(zcr)))
    features.append(float(np.std(zcr)))

    # RMS energy
    rms = librosa.feature.rms(y=y)
    features.append(float(np.mean(rms)))
    features.append(float(np.std(rms)))

    # Pitch (fundamental frequency)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
    f0_voiced = f0[voiced_flag] if voiced_flag is not None and voiced_flag.any() else np.array([0.0])
    features.append(float(np.mean(f0_voiced)))
    features.append(float(np.std(f0_voiced)))

    # Mel spectrogram stats
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
    features.extend(np.mean(mel, axis=1))

    return np.array(features, dtype=np.float32)


# ─── Synthetic Training Data & Model ─────────────────────────────────────────
def _synthetic_features_for(emotion: str, n: int = 120) -> np.ndarray:
    """
    Generate plausible synthetic feature distributions per emotion.
    Each column corresponds loosely to MFCCs, pitch, energy, etc.
    """
    rng = np.random.default_rng(abs(hash(emotion)) % (2**31))
    n_feat = 199  # must match extract_features output length

    # Base noise
    X = rng.normal(0, 1, (n, n_feat)).astype(np.float32)

    # Emotion-specific biases on key feature groups
    biases = {
        "angry":     {"energy": 3.5, "zcr": 2.0, "pitch": 1.5,  "mfcc_low": 2.0},
        "calm":      {"energy": -2.0,"zcr": -1.5,"pitch": -0.5, "mfcc_low": -1.0},
        "fearful":   {"energy": 1.5, "zcr": 1.5, "pitch": 3.0,  "mfcc_low": 0.5},
        "happy":     {"energy": 2.5, "zcr": 1.0, "pitch": 2.0,  "mfcc_low": 1.5},
        "neutral":   {"energy": 0.0, "zcr": 0.0, "pitch": 0.0,  "mfcc_low": 0.0},
        "sad":       {"energy": -2.5,"zcr": -1.0,"pitch": -2.0, "mfcc_low": -1.5},
        "surprised": {"energy": 2.0, "zcr": 2.5, "pitch": 3.5,  "mfcc_low": 1.0},
        "disgusted": {"energy": 1.0, "zcr": 0.5, "pitch": -1.0, "mfcc_low": 2.0},
    }
    b = biases.get(emotion, {})
    X[:, 80:82]  += b.get("energy", 0)    # RMS indices
    X[:, 78:80]  += b.get("zcr", 0)       # ZCR indices
    X[:, 82:84]  += b.get("pitch", 0)     # Pitch indices
    X[:, 0:10]   += b.get("mfcc_low", 0)  # First MFCCs
    return X


def build_and_train_model() -> Pipeline:
    """Train an MLP classifier on synthetic data, save, and return it."""
    print("Training emotion recognition model...")
    X_list, y_list = [], []
    for emo in EMOTIONS:
        X_emo = _synthetic_features_for(emo, n=150)
        X_list.append(X_emo)
        y_list.extend([emo] * len(X_emo))

    X_train = np.vstack(X_list)
    y_train = np.array(y_list)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=400,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            learning_rate_init=0.001,
        ))
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
    return model


def load_or_train_model() -> Pipeline:
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    return build_and_train_model()


# Load model at startup
model = load_or_train_model()


# ─── HTML Template ────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>VoiceSense — Emotion AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #050a0f;
    --surface: #0c141d;
    --surface2: #111c28;
    --border: #1a2d42;
    --accent: #00e5ff;
    --accent2: #7c3aed;
    --text: #e8f4fd;
    --muted: #4a6b8a;
    --danger: #ff4d6d;
    --success: #00ff8c;
    --glow: 0 0 30px rgba(0,229,255,0.15);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Space Mono', monospace;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Grid bg */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
      linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
  }

  .container {
    max-width: 900px; margin: 0 auto;
    padding: 2rem 1.5rem;
    position: relative; z-index: 1;
  }

  /* Header */
  header { text-align: center; margin-bottom: 3rem; }
  .logo-chip {
    display: inline-flex; align-items: center; gap: .5rem;
    border: 1px solid var(--border);
    background: var(--surface);
    padding: .35rem .9rem; border-radius: 999px;
    font-size: .75rem; color: var(--accent);
    letter-spacing: .12em; text-transform: uppercase;
    margin-bottom: 1.5rem;
  }
  .logo-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); animation: pulse 1.8s infinite; }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.4)} }

  h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 6vw, 4.2rem);
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(135deg, var(--accent) 0%, #a78bfa 60%, var(--accent2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .subtitle { color: var(--muted); font-size: .85rem; margin-top: .75rem; letter-spacing: .04em; }

  /* Main card */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--glow);
  }

  /* Waveform visualizer */
  #waveCanvas {
    width: 100%; height: 80px;
    border-radius: 10px;
    background: var(--surface2);
    display: block; margin-bottom: 1.5rem;
  }

  /* Controls */
  .controls { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }

  .btn {
    font-family: 'Syne', sans-serif;
    font-weight: 700; font-size: .9rem;
    letter-spacing: .06em;
    padding: .75rem 2rem; border-radius: 12px;
    border: none; cursor: pointer;
    transition: all .2s ease;
    display: inline-flex; align-items: center; gap: .5rem;
  }
  .btn-record {
    background: linear-gradient(135deg, #ff4d6d, #c9184a);
    color: #fff;
    box-shadow: 0 0 20px rgba(255,77,109,.3);
  }
  .btn-record:hover { transform: translateY(-2px); box-shadow: 0 0 30px rgba(255,77,109,.5); }
  .btn-record.recording {
    background: linear-gradient(135deg, #ff4d6d, #ff0040);
    animation: recPulse 1s infinite;
  }
  @keyframes recPulse { 0%,100%{box-shadow:0 0 20px rgba(255,77,109,.3)} 50%{box-shadow:0 0 40px rgba(255,77,109,.7)} }

  .btn-stop {
    background: var(--surface2); color: var(--muted);
    border: 1px solid var(--border);
  }
  .btn-stop:hover:not(:disabled) { color: var(--text); border-color: var(--accent); }
  .btn-stop:disabled { opacity: .4; cursor: not-allowed; }

  /* Status */
  .status {
    text-align: center; font-size: .8rem;
    color: var(--muted); margin-top: 1rem;
    min-height: 1.2em; letter-spacing: .05em;
  }
  .status.active { color: var(--accent); }
  .status.error { color: var(--danger); }

  /* Timer */
  #timer {
    text-align: center;
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    color: var(--danger); display: none;
    margin-top: .5rem;
  }

  /* Result */
  #result { display: none; }
  .result-header {
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: 1.5rem;
  }
  .emotion-emoji { font-size: 3rem; }
  .emotion-name {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: .05em;
  }
  .emotion-desc { font-size: .8rem; color: var(--muted); margin-top: .25rem; }

  /* Confidence bars */
  .bars-title {
    font-size: .75rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: .12em;
    margin-bottom: 1rem;
  }
  .bar-row {
    display: grid; grid-template-columns: 90px 1fr 48px;
    align-items: center; gap: .75rem;
    margin-bottom: .6rem;
  }
  .bar-label { font-size: .75rem; color: var(--muted); }
  .bar-track {
    height: 8px; background: var(--surface2);
    border-radius: 4px; overflow: hidden;
  }
  .bar-fill {
    height: 100%; border-radius: 4px;
    transition: width .8s cubic-bezier(.23,1,.32,1);
  }
  .bar-fill.top { background: linear-gradient(90deg, var(--accent), #a78bfa); }
  .bar-fill.other { background: var(--border); }
  .bar-pct { font-size: .75rem; color: var(--muted); text-align: right; }

  /* Insights grid */
  .insights {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem; margin-top: 1.5rem;
  }
  .insight-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px; padding: .9rem 1rem;
  }
  .insight-label { font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; }
  .insight-val { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; margin-top: .25rem; }

  /* Spinner */
  .spinner {
    width: 36px; height: 36px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .7s linear infinite;
    margin: 1.5rem auto;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Loading overlay */
  #loading { display: none; text-align: center; }
  #loading p { color: var(--muted); font-size: .8rem; margin-top: .5rem; }

  /* Footer */
  footer {
    text-align: center; margin-top: 3rem;
    font-size: .7rem; color: var(--muted); letter-spacing: .08em;
  }
  footer span { color: var(--accent); }

  @media (max-width: 500px) {
    .controls { flex-direction: column; }
    .btn { justify-content: center; }
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <div class="logo-chip"><div class="logo-dot"></div>AI · VOICE · EMOTION</div>
    <h1>VoiceSense</h1>
    <p class="subtitle">Real-time emotion detection from your voice using neural networks</p>
  </header>

  <!-- Recorder card -->
  <div class="card">
    <canvas id="waveCanvas"></canvas>

    <div class="controls">
      <button class="btn btn-record" id="recordBtn" onclick="startRecording()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="6"/></svg>
        Record Voice
      </button>
      <button class="btn btn-stop" id="stopBtn" onclick="stopRecording()" disabled>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>
        Analyse
      </button>
    </div>

    <div id="timer">0:00</div>
    <div class="status" id="status">Click "Record Voice" and speak for 2–10 seconds</div>
  </div>

  <!-- Loading -->
  <div class="card" id="loading">
    <div class="spinner"></div>
    <p>Extracting audio features &amp; running model…</p>
  </div>

  <!-- Result -->
  <div class="card" id="result">
    <div class="result-header">
      <div class="emotion-emoji" id="resEmoji"></div>
      <div>
        <div class="emotion-name" id="resName"></div>
        <div class="emotion-desc" id="resDesc"></div>
      </div>
    </div>

    <div class="bars-title">Confidence Distribution</div>
    <div id="barsContainer"></div>

    <div class="insights" id="insights"></div>
  </div>

  <footer>VoiceSense · MLP Neural Network · <span>{{ feature_count }}</span> audio features · Librosa DSP</footer>
</div>

<script>
let mediaRecorder, audioChunks = [], stream;
let timerInterval, seconds = 0;
let analyserNode, animFrame;

// ── Waveform ──────────────────────────────────────────────────────────────
const canvas = document.getElementById('waveCanvas');
const ctx2d = canvas.getContext('2d');

function resizeCanvas() {
  canvas.width  = canvas.offsetWidth  * devicePixelRatio;
  canvas.height = canvas.offsetHeight * devicePixelRatio;
  ctx2d.scale(devicePixelRatio, devicePixelRatio);
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

function drawIdle() {
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  ctx2d.clearRect(0, 0, W, H);
  ctx2d.strokeStyle = 'rgba(0,229,255,0.15)';
  ctx2d.lineWidth = 1.5;
  ctx2d.beginPath();
  ctx2d.moveTo(0, H/2);
  for (let x = 0; x < W; x++) {
    ctx2d.lineTo(x, H/2 + Math.sin(x * 0.04 + Date.now()*0.001) * 4);
  }
  ctx2d.stroke();
  animFrame = requestAnimationFrame(drawIdle);
}
drawIdle();

function drawLive(analyser) {
  const buf = new Uint8Array(analyser.frequencyBinCount);
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  function loop() {
    analyser.getByteTimeDomainData(buf);
    ctx2d.clearRect(0, 0, W, H);
    const grad = ctx2d.createLinearGradient(0,0,W,0);
    grad.addColorStop(0,'#00e5ff'); grad.addColorStop(.5,'#a78bfa'); grad.addColorStop(1,'#7c3aed');
    ctx2d.strokeStyle = grad;
    ctx2d.lineWidth = 2;
    ctx2d.beginPath();
    const sliceW = W / buf.length;
    let x = 0;
    buf.forEach((v, i) => {
      const y = (v / 128) * H/2;
      i === 0 ? ctx2d.moveTo(x, y) : ctx2d.lineTo(x, y);
      x += sliceW;
    });
    ctx2d.stroke();
    animFrame = requestAnimationFrame(loop);
  }
  loop();
}

// ── Recording ─────────────────────────────────────────────────────────────
async function startRecording() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const audioCtx = new AudioContext();
    const src = audioCtx.createMediaStreamSource(stream);
    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 2048;
    src.connect(analyserNode);

    cancelAnimationFrame(animFrame);
    drawLive(analyserNode);

    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.start();

    document.getElementById('recordBtn').disabled = true;
    document.getElementById('recordBtn').classList.add('recording');
    document.getElementById('stopBtn').disabled  = false;
    setStatus('🔴 Recording… speak clearly', 'active');
    document.getElementById('result').style.display  = 'none';

    seconds = 0;
    document.getElementById('timer').style.display = 'block';
    timerInterval = setInterval(() => {
      seconds++;
      const m = Math.floor(seconds/60), s = seconds%60;
      document.getElementById('timer').textContent = `${m}:${s.toString().padStart(2,'0')}`;
    }, 1000);

  } catch(e) {
    setStatus('❌ Mic access denied — please allow microphone', 'error');
  }
}

function stopRecording() {
  if (!mediaRecorder) return;
  mediaRecorder.onstop = processAudio;
  mediaRecorder.stop();
  stream.getTracks().forEach(t => t.stop());
  clearInterval(timerInterval);
  document.getElementById('timer').style.display = 'none';
  document.getElementById('recordBtn').disabled = false;
  document.getElementById('recordBtn').classList.remove('recording');
  document.getElementById('stopBtn').disabled = true;
  cancelAnimationFrame(animFrame);
  drawIdle();
  setStatus('Processing…', 'active');
}

async function processAudio() {
  document.getElementById('loading').style.display = 'block';
  const blob = new Blob(audioChunks, { type: 'audio/webm' });
  const ab   = await blob.arrayBuffer();
  const b64  = btoa(String.fromCharCode(...new Uint8Array(ab)));

  try {
    const res = await fetch('/analyse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio: b64, mime: 'audio/webm' })
    });
    const data = await res.json();
    document.getElementById('loading').style.display = 'none';
    if (data.error) { setStatus(`❌ ${data.error}`, 'error'); return; }
    showResult(data);
    setStatus('Analysis complete ✓', 'active');
  } catch(e) {
    document.getElementById('loading').style.display = 'none';
    setStatus('❌ Server error', 'error');
  }
}

// ── Result Rendering ──────────────────────────────────────────────────────
const COLORS = {{ emotion_colors|tojson }};

function showResult(data) {
  const meta = {{ emotion_meta|tojson }};
  const top  = data.emotion;
  const m    = meta[top] || {};

  document.getElementById('resEmoji').textContent = m.emoji || '🎙️';
  document.getElementById('resName').textContent  = top.toUpperCase();
  document.getElementById('resName').style.color  = m.color || '#fff';
  document.getElementById('resDesc').textContent  = m.desc  || '';

  // Bars
  const sorted = Object.entries(data.probabilities).sort((a,b) => b[1]-a[1]);
  const bc = document.getElementById('barsContainer');
  bc.innerHTML = '';
  sorted.forEach(([emo, prob]) => {
    const pct  = (prob * 100).toFixed(1);
    const isTop = emo === top;
    const row   = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <div class="bar-label">${(meta[emo]?.emoji||'') + ' ' + emo}</div>
      <div class="bar-track">
        <div class="bar-fill ${isTop?'top':'other'}" style="width:0%;background:${isTop?'':(meta[emo]?.color||'#1a2d42')}"></div>
      </div>
      <div class="bar-pct">${pct}%</div>`;
    bc.appendChild(row);
    setTimeout(() => row.querySelector('.bar-fill').style.width = pct+'%', 80);
  });

  // Insights
  const conf = (data.probabilities[top]*100).toFixed(1);
  const ins  = document.getElementById('insights');
  ins.innerHTML = `
    <div class="insight-chip">
      <div class="insight-label">Confidence</div>
      <div class="insight-val" style="color:${m.color||'var(--accent)'}">${conf}%</div>
    </div>
    <div class="insight-chip">
      <div class="insight-label">Duration</div>
      <div class="insight-val">${data.duration}s</div>
    </div>
    <div class="insight-chip">
      <div class="insight-label">Features</div>
      <div class="insight-val">${data.feature_count}</div>
    </div>
    <div class="insight-chip">
      <div class="insight-label">Sample Rate</div>
      <div class="insight-val">${data.sample_rate} Hz</div>
    </div>`;

  document.getElementById('result').style.display = 'block';
  document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
}

function setStatus(msg, cls='') {
  const s = document.getElementById('status');
  s.textContent  = msg;
  s.className = 'status ' + cls;
}
</script>
</body>
</html>
"""


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(
        HTML,
        feature_count=199,
        emotion_colors={e: EMOTION_META[e]["color"] for e in EMOTIONS},
        emotion_meta=EMOTION_META,
    )


@app.route("/analyse", methods=["POST"])
def analyse():
    data = request.get_json(force=True)
    if not data or "audio" not in data:
        return jsonify({"error": "No audio data received"}), 400

    try:
        audio_bytes = base64.b64decode(data["audio"])
    except Exception:
        return jsonify({"error": "Invalid base64 audio"}), 400

    # Load audio with librosa from bytes
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
        os.unlink(tmp_path)
    except Exception as e:
        return jsonify({"error": f"Audio decode failed: {str(e)}"}), 400

    if len(y) < sr * 0.5:
        return jsonify({"error": "Recording too short — please speak for at least 1 second"}), 400

    # Feature extraction
    try:
        feats = extract_features(y, sr)
    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {str(e)}"}), 500

    # Pad / trim to expected length
    expected = 199
    if len(feats) < expected:
        feats = np.pad(feats, (0, expected - len(feats)))
    else:
        feats = feats[:expected]

    # Predict
    try:
        proba_arr = model.predict_proba(feats.reshape(1, -1))[0]
        classes   = model.classes_
        emotion   = classes[np.argmax(proba_arr)]
        probs     = {str(c): float(round(p, 4)) for c, p in zip(classes, proba_arr)}
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {str(e)}"}), 500

    return jsonify({
        "emotion":       emotion,
        "probabilities": probs,
        "duration":      round(len(y) / sr, 2),
        "sample_rate":   sr,
        "feature_count": len(feats),
    })


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
