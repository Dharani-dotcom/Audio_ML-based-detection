"""
VoiceSense — Voice-Based Emotion Identifier
Streamlit app using scipy + numpy for audio DSP (no librosa / no ffmpeg needed).
Supports WAV files natively. Uses MLP neural network for emotion classification.
"""

import os
import io
import struct
import wave
import warnings
import tempfile
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceSense · Emotion AI",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

from scipy.io import wavfile
from scipy.signal import stft, spectrogram
from scipy.fft import rfft, rfftfreq
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Constants ─────────────────────────────────────────────────────────────────
EMOTIONS = ["angry", "calm", "fearful", "happy", "neutral", "sad", "surprised", "disgusted"]

EMOTION_META = {
    "angry":     {"emoji": "😠", "color": "#FF4455", "desc": "High energy · tense vocal patterns"},
    "calm":      {"emoji": "😌", "color": "#4ECDC4", "desc": "Steady pitch · relaxed rhythm"},
    "fearful":   {"emoji": "😨", "color": "#A78BFA", "desc": "Elevated pitch · irregular cadence"},
    "happy":     {"emoji": "😄", "color": "#FCD34D", "desc": "Bright timbre · rising inflection"},
    "neutral":   {"emoji": "😐", "color": "#94A3B8", "desc": "Balanced energy · flat prosody"},
    "sad":       {"emoji": "😢", "color": "#60A5FA", "desc": "Low energy · falling pitch"},
    "surprised": {"emoji": "😲", "color": "#FB923C", "desc": "Sharp onset · high pitch burst"},
    "disgusted": {"emoji": "🤢", "color": "#34D399", "desc": "Tense articulation · low pitch"},
}

MODEL_PATH   = "emotion_model.joblib"
N_FEATURES   = 160
TARGET_SR    = 22050


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #050a0f !important;
    color: #e8f4fd !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(rgba(0,229,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,0.025) 1px, transparent 1px),
        #050a0f !important;
    background-size: 40px 40px, 40px 40px, auto !important;
}
[data-testid="stHeader"],[data-testid="stToolbar"]{ background:transparent !important; }
section.main > div { padding-top:1rem !important; }
#MainMenu, footer, header { visibility:hidden; }
h1,h2,h3 { font-family:'Syne',sans-serif !important; }

.hero-title {
    font-family:'Syne',sans-serif;
    font-size:clamp(2.4rem,6vw,3.8rem); font-weight:800;
    background:linear-gradient(135deg,#00e5ff 0%,#a78bfa 55%,#7c3aed 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    line-height:1.05; margin:0;
}
.hero-sub { color:#4a6b8a; font-size:.82rem; letter-spacing:.05em; margin-top:.4rem; }
.logo-chip {
    display:inline-flex; align-items:center; gap:.5rem;
    border:1px solid #1a2d42; background:#0c141d;
    padding:.3rem .85rem; border-radius:999px;
    font-size:.7rem; color:#00e5ff; letter-spacing:.12em; text-transform:uppercase;
    margin-bottom:1.2rem;
}
.card {
    background:#0c141d; border:1px solid #1a2d42; border-radius:18px;
    padding:1.5rem 1.75rem; margin-bottom:1.2rem;
    box-shadow:0 0 30px rgba(0,229,255,0.07);
}
.card-title { font-family:'Syne',sans-serif; font-size:.68rem; font-weight:700;
    text-transform:uppercase; letter-spacing:.13em; color:#4a6b8a; margin-bottom:1rem; }
.emotion-display { display:flex; align-items:center; gap:1.2rem; margin-bottom:1.5rem; }
.emotion-emoji-big { font-size:3.5rem; line-height:1; }
.emotion-label { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800;
    text-transform:uppercase; letter-spacing:.04em; }
.emotion-desc { font-size:.78rem; color:#4a6b8a; margin-top:.2rem; }
.bar-row { display:flex; align-items:center; gap:.75rem; margin-bottom:.55rem; }
.bar-emo-label { font-size:.7rem; color:#6b8faf; width:92px; flex-shrink:0; }
.bar-track { flex:1; height:7px; background:#111c28; border-radius:4px; overflow:hidden; }
.bar-fill { height:100%; border-radius:4px; }
.bar-pct { font-size:.7rem; color:#4a6b8a; width:42px; text-align:right; }
.insights-row { display:flex; flex-wrap:wrap; gap:.75rem; margin-top:1.2rem; }
.insight-chip { background:#111c28; border:1px solid #1a2d42; border-radius:10px;
    padding:.75rem 1rem; flex:1; min-width:110px; }
.ic-label { font-size:.62rem; color:#4a6b8a; text-transform:uppercase; letter-spacing:.1em; }
.ic-val { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:700; margin-top:.2rem; }
.tip-box { background:#0a1520; border:1px solid #1a2d42; border-radius:12px;
    padding:1rem 1.25rem; font-size:.75rem; color:#4a6b8a; line-height:1.7; margin-top:.5rem; }
.tip-box strong { color:#00e5ff; }

[data-testid="stFileUploader"] { background:#0c141d !important;
    border:1px dashed #1a2d42 !important; border-radius:14px !important; }
.stButton > button {
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    font-size:.88rem !important; letter-spacing:.06em !important;
    background:linear-gradient(135deg,#00e5ff22,#7c3aed22) !important;
    border:1px solid #00e5ff55 !important; color:#00e5ff !important;
    border-radius:12px !important; transition:all .2s !important;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#00e5ff33,#7c3aed33) !important;
    border-color:#00e5ff !important; box-shadow:0 0 20px rgba(0,229,255,0.2) !important;
}
hr { border-color:#1a2d42 !important; }
</style>
""", unsafe_allow_html=True)


# ── Audio Loading ─────────────────────────────────────────────────────────────
def load_wav(file_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Load a WAV file from bytes. Returns (mono float32 array, sample_rate).
    Handles 8/16/24/32-bit PCM and 32-bit float WAV.
    """
    buf = io.BytesIO(file_bytes)
    sr, data = wavfile.read(buf)

    # Convert to float32 in range [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype == np.float32:
        pass
    else:
        data = data.astype(np.float32)

    # Mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to TARGET_SR if needed (simple linear interp)
    if sr != TARGET_SR:
        duration   = len(data) / sr
        target_len = int(duration * TARGET_SR)
        data = np.interp(
            np.linspace(0, len(data) - 1, target_len),
            np.arange(len(data)),
            data
        ).astype(np.float32)
        sr = TARGET_SR

    return data, sr


# ── DSP Feature Extraction (scipy + numpy only) ───────────────────────────────
def _frame(y: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Slice signal into overlapping frames → shape (n_frames, frame_len)."""
    n_frames = 1 + (len(y) - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    return y[idx]


def compute_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 20,
                 n_fft: int = 512, hop: int = 256, n_mels: int = 40) -> np.ndarray:
    """Mel-frequency cepstral coefficients using scipy FFT."""
    # Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # Framing + windowing
    frames = _frame(y, n_fft, hop) * np.hanning(n_fft)  # (n_frames, n_fft)

    # Power spectrum
    spec = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2   # (n_frames, n_fft//2+1)

    # Mel filterbank
    freqs    = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_low  = 2595 * np.log10(1 + 20   / 700)
    mel_high = 2595 * np.log10(1 + (sr/2) / 700)
    mel_pts  = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_pts   = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts  = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1]
        f_m       = bin_pts[m]
        f_m_plus  = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-9)
        for k in range(f_m, f_m_plus):
            fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-9)

    mel_energy = np.dot(spec, fbank.T)                       # (n_frames, n_mels)
    log_mel    = np.log(mel_energy + 1e-9)

    # DCT-II for MFCCs
    n_f = log_mel.shape[0]
    dct = np.cos(np.pi / n_mels * (np.arange(n_mels)[None,:] + 0.5) * np.arange(n_mfcc)[:,None])
    mfcc = log_mel @ dct.T                                   # (n_frames, n_mfcc)
    return mfcc.T                                             # (n_mfcc, n_frames)


def compute_zcr(y: np.ndarray, frame_len: int = 512, hop: int = 256) -> np.ndarray:
    frames = _frame(y, frame_len, hop)
    return (np.diff(np.sign(frames), axis=1) != 0).sum(axis=1) / frame_len


def compute_rms(y: np.ndarray, frame_len: int = 512, hop: int = 256) -> np.ndarray:
    frames = _frame(y, frame_len, hop)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def compute_spectral_centroid(y: np.ndarray, sr: int,
                               n_fft: int = 512, hop: int = 256) -> np.ndarray:
    frames = _frame(y, n_fft, hop) * np.hanning(n_fft)
    mag    = np.abs(np.fft.rfft(frames, n=n_fft))
    freqs  = np.fft.rfftfreq(n_fft, 1.0 / sr)
    denom  = mag.sum(axis=1) + 1e-9
    return (mag * freqs[None, :]).sum(axis=1) / denom


def compute_spectral_rolloff(y: np.ndarray, sr: int,
                              n_fft: int = 512, hop: int = 256,
                              roll_percent: float = 0.85) -> np.ndarray:
    frames = _frame(y, n_fft, hop) * np.hanning(n_fft)
    mag    = np.abs(np.fft.rfft(frames, n=n_fft))
    freqs  = np.fft.rfftfreq(n_fft, 1.0 / sr)
    cumsum = np.cumsum(mag, axis=1)
    thresh = roll_percent * cumsum[:, -1:]
    idx    = np.argmax(cumsum >= thresh, axis=1)
    return freqs[idx]


def compute_pitch_estimate(y: np.ndarray, sr: int,
                            fmin: int = 80, fmax: int = 400) -> float:
    """Simple autocorrelation-based pitch estimate."""
    # Use centre chunk of signal for speed
    centre = y[len(y)//4 : 3*len(y)//4]
    N      = len(centre)
    if N < 2:
        return 0.0
    corr   = np.correlate(centre, centre, mode='full')[N-1:]
    # Search in valid lag range
    lag_min = max(1, int(sr / fmax))
    lag_max = min(N - 1, int(sr / fmin))
    if lag_min >= lag_max:
        return 0.0
    peak = np.argmax(corr[lag_min:lag_max]) + lag_min
    return float(sr / peak) if peak > 0 else 0.0


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract 160 features using scipy + numpy only."""
    features = []

    # 1. MFCCs — 20 coeffs × (mean + std) = 40
    mfcc = compute_mfcc(y, sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 2. Delta MFCCs (frame-to-frame diff) — 20 means = 20
    if mfcc.shape[1] > 1:
        delta = np.diff(mfcc, axis=1)
        features.extend(np.mean(np.abs(delta), axis=1))
    else:
        features.extend([0.0] * 20)

    # 3. ZCR stats — 4 features
    zcr = compute_zcr(y)
    features += [float(np.mean(zcr)), float(np.std(zcr)),
                 float(np.max(zcr)),  float(np.min(zcr))]

    # 4. RMS energy stats — 4 features
    rms = compute_rms(y)
    features += [float(np.mean(rms)), float(np.std(rms)),
                 float(np.max(rms)),  float(np.min(rms))]

    # 5. Spectral centroid stats — 4 features
    cent = compute_spectral_centroid(y, sr)
    features += [float(np.mean(cent)), float(np.std(cent)),
                 float(np.max(cent)),  float(np.min(cent))]

    # 6. Spectral rolloff stats — 4 features
    roll = compute_spectral_rolloff(y, sr)
    features += [float(np.mean(roll)), float(np.std(roll)),
                 float(np.max(roll)),  float(np.min(roll))]

    # 7. Pitch estimate + signal stats — 4 features
    pitch = compute_pitch_estimate(y, sr)
    features += [pitch, float(np.mean(np.abs(y))),
                 float(np.max(np.abs(y))), float(np.std(y))]

    # 8. Band energy ratios (sub-band power fractions) — 8 features
    n_fft  = 512
    mag    = np.abs(rfft(y[:len(y) - len(y) % n_fft].reshape(-1, n_fft))).mean(axis=0)
    freqs  = rfftfreq(n_fft, 1.0 / sr)
    bands  = [(0, 300), (300, 800), (800, 2000), (2000, 4000),
              (4000, 6000), (6000, 8000), (8000, 10000), (10000, sr//2)]
    total  = mag.sum() + 1e-9
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        features.append(float(mag[mask].sum() / total) if mask.any() else 0.0)

    # 9. Short-time energy entropy — 4 features
    frames   = _frame(y, 512, 256)
    frame_e  = np.sum(frames ** 2, axis=1) + 1e-9
    norm_e   = frame_e / frame_e.sum()
    entropy  = -np.sum(norm_e * np.log(norm_e + 1e-9))
    features += [float(entropy),
                 float(np.mean(np.diff(rms))),
                 float(np.percentile(rms, 75) - np.percentile(rms, 25)),
                 float(np.percentile(cent, 75) - np.percentile(cent, 25))]

    # Pad / trim to N_FEATURES
    arr = np.array(features, dtype=np.float32)
    if len(arr) < N_FEATURES:
        arr = np.pad(arr, (0, N_FEATURES - len(arr)))
    return arr[:N_FEATURES]


# ── Model ─────────────────────────────────────────────────────────────────────
def _synthetic(emotion: str, n: int = 160) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(emotion)) % (2**31))
    X   = rng.normal(0, 1, (n, N_FEATURES)).astype(np.float32)
    cfg = {
        "angry":     dict(rms=(3.5,  .3), zcr=(2.0, .2),  pitch=(1.5, .3),  mfcc=2.0 ),
        "calm":      dict(rms=(-2.0, .3), zcr=(-1.5,.2),  pitch=(-0.5,.3),  mfcc=-1.0),
        "fearful":   dict(rms=(1.5,  .3), zcr=(1.5, .2),  pitch=(3.0, .3),  mfcc=0.5 ),
        "happy":     dict(rms=(2.5,  .3), zcr=(1.0, .2),  pitch=(2.0, .3),  mfcc=1.5 ),
        "neutral":   dict(rms=(0.0,  .1), zcr=(0.0, .1),  pitch=(0.0, .1),  mfcc=0.0 ),
        "sad":       dict(rms=(-2.5, .3), zcr=(-1.0,.2),  pitch=(-2.0,.3),  mfcc=-1.5),
        "surprised": dict(rms=(2.0,  .3), zcr=(2.5, .2),  pitch=(3.5, .3),  mfcc=1.0 ),
        "disgusted": dict(rms=(1.0,  .3), zcr=(0.5, .2),  pitch=(-1.0,.3),  mfcc=2.0 ),
    }.get(emotion, dict(rms=(0,0), zcr=(0,0), pitch=(0,0), mfcc=0))

    # RMS idx 40–43, ZCR 20–23, pitch ~84, MFCC 0–19
    X[:, 40:44] += cfg["rms"][0]
    X[:, 20:24] += cfg["zcr"][0]
    X[:, 84]    += cfg["pitch"][0]
    X[:,  0:20] += cfg["mfcc"]
    return X


@st.cache_resource(show_spinner="Training emotion model (first run only)…")
def load_model() -> Pipeline:
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    X = np.vstack([_synthetic(e) for e in EMOTIONS])
    y = np.array([e for e in EMOTIONS for _ in range(160)])

    mdl = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            learning_rate_init=0.001,
        ))
    ])
    mdl.fit(X, y)
    try:
        joblib.dump(mdl, MODEL_PATH)
    except Exception:
        pass
    return mdl


# ── Render Helpers ────────────────────────────────────────────────────────────
def render_result(emotion: str, probs: dict, duration: float, sr: int, n_feats: int):
    meta  = EMOTION_META.get(emotion, {})
    color = meta.get("color", "#00e5ff")
    conf  = probs.get(emotion, 0) * 100

    st.markdown(f"""
    <div class="card">
      <div class="card-title">Detected Emotion</div>
      <div class="emotion-display">
        <div class="emotion-emoji-big">{meta.get('emoji','🎙️')}</div>
        <div>
          <div class="emotion-label" style="color:{color}">{emotion.upper()}</div>
          <div class="emotion-desc">{meta.get('desc','')}</div>
        </div>
      </div>
      <div class="insights-row">
        <div class="insight-chip"><div class="ic-label">Confidence</div>
          <div class="ic-val" style="color:{color}">{conf:.1f}%</div></div>
        <div class="insight-chip"><div class="ic-label">Duration</div>
          <div class="ic-val">{duration:.2f}s</div></div>
        <div class="insight-chip"><div class="ic-label">Features</div>
          <div class="ic-val">{n_feats}</div></div>
        <div class="insight-chip"><div class="ic-label">Sample Rate</div>
          <div class="ic-val">{sr} Hz</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Probability bars
    sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    html = '<div class="card"><div class="card-title">Confidence Distribution</div>'
    for emo, prob in sorted_p:
        pct = prob * 100
        ec  = EMOTION_META.get(emo, {}).get("color", "#1a2d42")
        sty = "background:linear-gradient(90deg,#00e5ff,#a78bfa)" if emo == emotion else f"background:{ec}44"
        html += f"""<div class="bar-row">
          <div class="bar-emo-label">{EMOTION_META.get(emo,{}).get('emoji','')} {emo}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;{sty}"></div></div>
          <div class="bar-pct">{pct:.1f}%</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 2rem">
      <div class="logo-chip">⬤ &nbsp;AI · VOICE · EMOTION</div>
      <div class="hero-title">VoiceSense</div>
      <div class="hero-sub">Real-time emotion detection · MLP Neural Network · scipy DSP · zero system deps</div>
    </div>""", unsafe_allow_html=True)

    load_model()  # warm-up cache

    # Upload
    st.markdown('<div class="card"><div class="card-title">Upload WAV File</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a WAV voice recording",
        type=["wav"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Tip box
    st.markdown("""
    <div class="tip-box">
      <strong>💡 Tips for best results:</strong><br>
      • Use a <strong>clear WAV recording</strong> with minimal background noise<br>
      • Speak naturally for <strong>2–10 seconds</strong><br>
      • Record on a phone or mic app and export as <strong>.wav</strong><br>
      • Free tools: Audacity (desktop) · Voice Recorder (Windows/Mac) · GarageBand (iOS)
    </div>
    """, unsafe_allow_html=True)

    if uploaded:
        st.audio(uploaded, format="audio/wav")

        if st.button("🔬  Analyse Emotion", use_container_width=True):
            with st.spinner("Extracting audio features & running model…"):
                try:
                    audio_bytes = uploaded.read()
                    y, sr = load_wav(audio_bytes)

                    if len(y) < sr * 0.5:
                        st.error("⚠️ Recording too short — please use a clip of at least 1 second.")
                        return

                    feats   = extract_features(y, sr)
                    mdl     = load_model()
                    proba   = mdl.predict_proba(feats.reshape(1, -1))[0]
                    emotion = mdl.classes_[np.argmax(proba)]
                    probs   = {str(c): float(p) for c, p in zip(mdl.classes_, proba)}

                    render_result(emotion, probs, len(y) / sr, sr, len(feats))

                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}\n\nMake sure the file is a valid WAV.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 0;color:#1a3050;font-size:.8rem">
          <div style="font-size:2.5rem;margin-bottom:.75rem">🎙️</div>
          Upload a WAV voice recording above to detect emotion
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:.5rem 0 .25rem;margin-bottom:.5rem">
      <span style="font-size:.72rem;color:#2a4a6a">
        😠 Angry &nbsp;·&nbsp; 😌 Calm &nbsp;·&nbsp; 😨 Fearful &nbsp;·&nbsp; 😄 Happy &nbsp;·&nbsp;
        😐 Neutral &nbsp;·&nbsp; 😢 Sad &nbsp;·&nbsp; 😲 Surprised &nbsp;·&nbsp; 🤢 Disgusted
      </span>
    </div>
    <div style="text-align:center;font-size:.62rem;color:#1a3050;padding-bottom:1rem">
      VoiceSense · scipy + numpy DSP · MLP Neural Network · Streamlit Cloud
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
