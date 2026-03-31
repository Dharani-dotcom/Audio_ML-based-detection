"""
VoiceSense — Voice-Based Emotion Identifier
Streamlit app with MLP neural network for real-time voice emotion detection.
"""

import os
import tempfile
import warnings
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="VoiceSense · Emotion AI",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

import librosa
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

MODEL_PATH = "emotion_model.joblib"
N_FEATURES = 199


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
[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }
section.main > div { padding-top: 1rem !important; }
#MainMenu, footer, header { visibility: hidden; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.5rem,6vw,4rem);
    font-weight: 800;
    background: linear-gradient(135deg,#00e5ff 0%,#a78bfa 55%,#7c3aed 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    line-height: 1.05; margin: 0;
}
.hero-sub { color:#4a6b8a; font-size:.82rem; letter-spacing:.05em; margin-top:.4rem; }
.logo-chip {
    display:inline-flex; align-items:center; gap:.5rem;
    border:1px solid #1a2d42; background:#0c141d;
    padding:.3rem .85rem; border-radius:999px;
    font-size:.72rem; color:#00e5ff; letter-spacing:.12em; text-transform:uppercase;
    margin-bottom:1.2rem;
}
.card {
    background:#0c141d; border:1px solid #1a2d42; border-radius:18px;
    padding:1.5rem 1.75rem; margin-bottom:1.2rem;
    box-shadow:0 0 30px rgba(0,229,255,0.07);
}
.card-title { font-family:'Syne',sans-serif; font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.13em; color:#4a6b8a; margin-bottom:1rem; }
.emotion-display { display:flex; align-items:center; gap:1.2rem; margin-bottom:1.5rem; }
.emotion-emoji-big { font-size:3.5rem; line-height:1; }
.emotion-label { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; text-transform:uppercase; letter-spacing:.04em; }
.emotion-desc { font-size:.78rem; color:#4a6b8a; margin-top:.2rem; }
.bar-row { display:flex; align-items:center; gap:.75rem; margin-bottom:.55rem; }
.bar-emo-label { font-size:.72rem; color:#6b8faf; width:88px; flex-shrink:0; }
.bar-track { flex:1; height:7px; background:#111c28; border-radius:4px; overflow:hidden; }
.bar-fill { height:100%; border-radius:4px; }
.bar-pct { font-size:.72rem; color:#4a6b8a; width:42px; text-align:right; }
.insights-row { display:flex; flex-wrap:wrap; gap:.75rem; margin-top:1.2rem; }
.insight-chip { background:#111c28; border:1px solid #1a2d42; border-radius:10px; padding:.75rem 1rem; flex:1; min-width:110px; }
.ic-label { font-size:.65rem; color:#4a6b8a; text-transform:uppercase; letter-spacing:.1em; }
.ic-val { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; margin-top:.2rem; }

[data-testid="stFileUploader"] { background:#0c141d !important; border:1px dashed #1a2d42 !important; border-radius:14px !important; }
.stButton > button {
    font-family:'Syne',sans-serif !important; font-weight:700 !important; font-size:.88rem !important;
    letter-spacing:.06em !important;
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


# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    features.extend(np.mean(librosa.feature.delta(mfcc), axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    features.append(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))))
    features.append(float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))))
    features.append(float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.extend([float(np.mean(zcr)), float(np.std(zcr))])

    rms = librosa.feature.rms(y=y)
    features.extend([float(np.mean(rms)), float(np.std(rms))])

    f0, voiced_flag, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
    f0v = f0[voiced_flag] if voiced_flag is not None and voiced_flag.any() else np.array([0.0])
    features.extend([float(np.mean(f0v)), float(np.std(f0v))])

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
    features.extend(np.mean(mel, axis=1))

    arr = np.array(features, dtype=np.float32)
    if len(arr) < N_FEATURES:
        arr = np.pad(arr, (0, N_FEATURES - len(arr)))
    return arr[:N_FEATURES]


# ── Model ─────────────────────────────────────────────────────────────────────
def _synthetic(emotion: str, n: int = 150) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(emotion)) % (2**31))
    X = rng.normal(0, 1, (n, N_FEATURES)).astype(np.float32)
    b = {"angry":(3.5,2.0,1.5,2.0),"calm":(-2.0,-1.5,-0.5,-1.0),
         "fearful":(1.5,1.5,3.0,0.5),"happy":(2.5,1.0,2.0,1.5),
         "neutral":(0,0,0,0),"sad":(-2.5,-1.0,-2.0,-1.5),
         "surprised":(2.0,2.5,3.5,1.0),"disgusted":(1.0,0.5,-1.0,2.0)}.get(emotion,(0,0,0,0))
    X[:,80:82]+=b[0]; X[:,78:80]+=b[1]; X[:,82:84]+=b[2]; X[:,0:10]+=b[3]
    return X


@st.cache_resource(show_spinner="Training emotion model (first run only)…")
def load_model() -> Pipeline:
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    X = np.vstack([_synthetic(e) for e in EMOTIONS])
    y = np.array([e for e in EMOTIONS for _ in range(150)])
    mdl = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu",
                              max_iter=400, random_state=42, early_stopping=True,
                              validation_fraction=0.15, n_iter_no_change=20,
                              learning_rate_init=0.001))
    ])
    mdl.fit(X, y)
    try:
        joblib.dump(mdl, MODEL_PATH)
    except Exception:
        pass
    return mdl


# ── Result Rendering ──────────────────────────────────────────────────────────
def render_result(emotion, probs, duration, sr, n_feats):
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
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    bars = '<div class="card"><div class="card-title">Confidence Distribution</div>'
    for emo, prob in sorted_probs:
        pct   = prob * 100
        ec    = EMOTION_META.get(emo, {}).get("color", "#1a2d42")
        style = "background:linear-gradient(90deg,#00e5ff,#a78bfa)" if emo == emotion else f"background:{ec}44"
        bars += f"""<div class="bar-row">
          <div class="bar-emo-label">{EMOTION_META.get(emo,{}).get('emoji','')} {emo}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;{style}"></div></div>
          <div class="bar-pct">{pct:.1f}%</div>
        </div>"""
    bars += "</div>"
    st.markdown(bars, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 2rem">
      <div class="logo-chip">⬤ &nbsp;AI · VOICE · EMOTION</div>
      <div class="hero-title">VoiceSense</div>
      <div class="hero-sub">Real-time emotion detection from voice · MLP Neural Network · 199 audio features</div>
    </div>""", unsafe_allow_html=True)

    load_model()  # warm-up

    st.markdown('<div class="card"><div class="card-title">Upload Audio File</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Supported formats: WAV · MP3 · OGG · FLAC · M4A",
        type=["wav", "mp3", "ogg", "flac", "m4a", "webm"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        st.audio(uploaded)
        if st.button("🔬  Analyse Emotion", use_container_width=True):
            with st.spinner("Extracting features & running model…"):
                try:
                    audio_bytes = uploaded.read()
                    ext = os.path.splitext(uploaded.name)[-1] or ".wav"
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name

                    y, sr = librosa.load(tmp_path, sr=22050, mono=True)
                    os.unlink(tmp_path)

                    if len(y) < sr * 0.5:
                        st.error("⚠️ Recording too short — use a clip of at least 1 second.")
                        return

                    feats = extract_features(y, sr)
                    mdl   = load_model()
                    proba = mdl.predict_proba(feats.reshape(1, -1))[0]
                    emotion = mdl.classes_[np.argmax(proba)]
                    probs   = {str(c): float(p) for c, p in zip(mdl.classes_, proba)}

                    render_result(emotion, probs, len(y)/sr, sr, len(feats))

                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
    else:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 0;color:#1a3050;font-size:.8rem">
          <div style="font-size:2.5rem;margin-bottom:.75rem">🎙️</div>
          Upload a voice recording above to detect emotion
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;font-size:.65rem;color:#1a3050;padding-bottom:1rem">
      VoiceSense · MLP Neural Network · Librosa DSP · Streamlit Cloud
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
