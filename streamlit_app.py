import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import joblib
import io
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

st.set_page_config(page_title="Parkinson's Detection", layout="centered")

st.title("🎵 Parkinson's Disease Detection from Voice")
st.write("Upload a `.wav` audio file to determine if the speaker may have Parkinson’s Disease (PD) based on voice features.")

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Audio upload
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    # Load audio
    y, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file, format="audio/wav")

    # Display waveform
    st.subheader("Waveform")
    fig_wave, ax_wave = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax_wave)
    ax_wave.set_title("Audio Waveform")
    st.pyplot(fig_wave)

    # Display spectrogram
    st.subheader("Spectrogram")
    stft = librosa.stft(y)
    db_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    fig_spec, ax_spec = plt.subplots()
    img = librosa.display.specshow(db_stft, sr=sr, x_axis='time', y_axis='log', ax=ax_spec)
    ax_spec.set_title("Log-Frequency Spectrogram")
    fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
    st.pyplot(fig_spec)

    # Feature extraction
    def extract_features(y, sr):
        features = []

        # Basic statistical features
        features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
        features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

        # MFCCs (take first 13)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        features.extend(mfccs_mean)

        return np.array(features).reshape(1, -1)

    # Extract & scale features
    extracted = extract_features(y, sr)
    extracted_scaled = scaler.transform(extracted)

    # Prediction
    prediction = model.predict(extracted_scaled)[0]
    proba = model.predict_proba(extracted_scaled)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("Prediction")
    if prediction == 1:
        st.error("🧠 The model predicts: **Parkinson's Disease (PD)**")
    else:
        st.success("✅ The model predicts: **No Parkinson's Disease (Non-PD)**")

    if proba is not None:
        st.metric(label="PD Probability", value=f"{proba*100:.2f}%")
