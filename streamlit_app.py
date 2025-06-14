import streamlit as st
import numpy as np
import librosa
import joblib
import soundfile as sf
import os
import tempfile

# Title
st.title("🎤 Parkinson's Detection from Voice")
st.write("Upload a `.wav` file and let the model predict whether the voice indicates Parkinson’s.")

# Load trained model
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")  # Ensure this file exists in the app directory
    return model

model = load_model()

# Feature extractor
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = []

    # 1. MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # 2. Chroma
    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # 3. Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)

    # Combine features
    features = np.hstack([mfccs_mean, chroma_mean, mel_mean])
    return features.reshape(1, -1)

# Upload audio file
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Extract features
    try:
        features = extract_features(tmp_path)
        prediction = model.predict(features)[0]

        st.success(f"🧠 Prediction: {'Parkinson\'s Detected' if prediction == 1 else 'Healthy'}")
    except Exception as e:
        st.error(f"Error processing audio: {e}")

    # Cleanup
    os.remove(tmp_path)
