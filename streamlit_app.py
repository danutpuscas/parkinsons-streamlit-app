import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import joblib
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler

# Title
st.title("Parkinson's Disease Detection from Voice")

# Load models and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# Upload audio
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    # Load and display audio
    audio, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file, format='audio/wav')

    # Plot waveform
    st.subheader("Waveform")
    fig_wave, ax = plt.subplots()
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title('Audio Waveform')
    st.pyplot(fig_wave)

    # Plot spectrogram
    st.subheader("Spectrogram")
    stft = librosa.stft(audio)
    db_stft = librosa.amplitude_to_db(np.abs(stft))
    fig_spec, ax = plt.subplots()
    img = librosa.display.specshow(db_stft, sr=sr, x_axis='time', y_axis='log', ax=ax)
    fig_spec.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title('Spectrogram')
    st.pyplot(fig_spec)

    # Extract features
    st.subheader("Model Prediction")
    def extract_features(audio, sr):
        features = []
        features.append(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
        features.append(np.mean(librosa.feature.rms(y=audio)))
        features.append(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        features.extend(np.mean(mfccs, axis=1))
        return np.array(features).reshape(1, -1)

    try:
        features = extract_features(audio, sr)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0][1] if hasattr(model, 'predict_proba') else None

        st.write(f"**Prediction:** {'Parkinson's Detected' if prediction else 'No Parkinson's'}")
        if proba is not None:
            st.write(f"**Confidence:** {proba * 100:.2f}%")
    except Exception as e:
        st.error(f"An error occurred while processing the audio: {str(e)}")
else:
    st.info("Please upload a WAV file to get a prediction.")
