import streamlit as st
import numpy as np
import librosa
import joblib
import os
from streamlit_audiorec import st_audiorec
import soundfile as sf
import tempfile

# Load models
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Parkinson's Detection from Voice")

option = st.radio("Select input method:", ("Upload .wav file", "Record from microphone"))

uploaded_audio = None
if option == "Upload .wav file":
    uploaded_audio = st.file_uploader("Upload a .wav file", type=["wav"])

elif option == "Record from microphone":
    audio_bytes = st_audiorec()
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            uploaded_audio = temp_audio.name

if uploaded_audio:
    try:
        y, sr = librosa.load(uploaded_audio, sr=22050)

        # Extract features (20 features total to match training data)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        features = [
            np.mean(mfccs[:7]),  # first 7 MFCCs
            np.std(mfccs[:7]),
            np.mean(zcr),
            np.mean(centroid),
            np.mean(bandwidth),
            np.mean(contrast),
            np.std(centroid),
            np.std(bandwidth),
            np.std(contrast)
        ]

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]

        st.success("Prediction: PwPD" if pred == 1 else "Prediction: Healthy Control")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload or record a .wav file to start.")
