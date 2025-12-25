import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st

# ✅ FIRST Streamlit command
st.set_page_config(
    page_title="Speech Emotion Recognition",
    layout="centered"
)

import numpy as np
import librosa
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
SR = 16000
MAX_LEN = 200
N_MELS = 64

MODEL_PATH = r"C:\Users\Supratim\OneDrive\문서\DLRL Project\ser_combined_lstm.h5"
ENCODER_PATH = r"C:\Users\Supratim\OneDrive\문서\DLRL Project\label_encoder.pkl"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_ser():
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_ser()

# ---------------- FEATURE EXTRACTION ----------------
def extract_logmel(audio, sr):
    audio, _ = librosa.effects.trim(audio)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=2048,
        hop_length=512
    )

    logmel = librosa.power_to_db(mel)

    if logmel.shape[1] < MAX_LEN:
        logmel = np.pad(logmel, ((0,0),(0,MAX_LEN-logmel.shape[1])))
    else:
        logmel = logmel[:, :MAX_LEN]

    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-9)
    return logmel.T

# ---------------- UI ----------------
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a WAV file to detect emotion")

uploaded = st.file_uploader("Upload audio", type=["wav"])

if uploaded:
    audio, sr = librosa.load(uploaded, sr=SR, mono=True)
    st.audio(uploaded)

    if st.button("Predict Emotion"):
        features = extract_logmel(audio, sr)
        features = np.expand_dims(features, axis=0)

        probs = model.predict(features)[0]
        idx = np.argmax(probs)

        emotion = le.inverse_transform([idx])[0]
        confidence = probs[idx]

        st.success(f"Emotion: **{emotion.upper()}**")
        st.info(f"Confidence: **{confidence:.2f}**")

        fig, ax = plt.subplots()
        ax.bar(le.classes_, probs)
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        st.pyplot(fig)
