"""
Voice-Based Emotion Identifier (Streamlit Version)
"""

import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─── Emotion Labels ─────────────────────────────
EMOTIONS = ["angry", "calm", "fearful", "happy", "neutral", "sad", "surprised", "disgusted"]

MODEL_PATH = "emotion_model.joblib"

# ─── Feature Extraction ─────────────────────────
def extract_features(y, sr):
    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(mfcc_delta, axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(float(np.mean(spec_cent)))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(float(np.mean(spec_bw)))

    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(float(np.mean(spec_roll)))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(float(np.mean(zcr)))
    features.append(float(np.std(zcr)))

    rms = librosa.feature.rms(y=y)
    features.append(float(np.mean(rms)))
    features.append(float(np.std(rms)))

    f0, voiced_flag, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
    f0_voiced = f0[voiced_flag] if voiced_flag is not None and voiced_flag.any() else np.array([0.0])
    features.append(float(np.mean(f0_voiced)))
    features.append(float(np.std(f0_voiced)))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
    features.extend(np.mean(mel, axis=1))

    return np.array(features, dtype=np.float32)


# ─── Synthetic Data ─────────────────────────────
def synthetic_data(emotion, n=120):
    rng = np.random.default_rng(abs(hash(emotion)) % (2**31))
    X = rng.normal(0, 1, (n, 199)).astype(np.float32)
    return X


def train_model():
    X_list, y_list = [], []
    for emo in EMOTIONS:
        X = synthetic_data(emo, 150)
        X_list.append(X)
        y_list.extend([emo]*len(X))

    X_train = np.vstack(X_list)
    y_train = np.array(y_list)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300))
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_model()


model = load_model()

# ─── Streamlit UI ───────────────────────────────
st.set_page_config(page_title="Voice Emotion AI", layout="centered")

st.title("🎙️ Voice Emotion Identifier")
st.write("Upload a voice file and detect emotion using AI")

audio_file = st.file_uploader("Upload audio (.wav/.mp3)", type=["wav", "mp3"])

if audio_file:
    st.audio(audio_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path, sr=22050)
    os.unlink(tmp_path)

    if st.button("Analyse Emotion"):
        with st.spinner("Processing..."):
            feats = extract_features(y, sr)

            if len(feats) < 199:
                feats = np.pad(feats, (0, 199-len(feats)))
            else:
                feats = feats[:199]

            proba = model.predict_proba(feats.reshape(1,-1))[0]
            classes = model.classes_

            result = classes[np.argmax(proba)]

        st.success(f"Detected Emotion: **{result.upper()}**")

        st.subheader("Confidence")
        for emo, p in zip(classes, proba):
            st.write(f"{emo}: {round(p*100,2)}%")
