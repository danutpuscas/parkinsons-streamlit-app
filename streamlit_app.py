import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
import tempfile
import os
import soundfile as sf
import base64
from io import BytesIO
from streamlit.components.v1 import html

# Optional microphone recording (streamlit-audiorec)
try:
    from streamlit_audiorec import st_audiorec
    mic_available = True
except ImportError:
    mic_available = False

st.set_page_config(page_title="🧠 Parkinson's Detection", layout="centered")
st.title("🧠 Parkinson's Detection from Voice")
st.write("Upload or record a `.wav` file of a sustained vowel sound (e.g., 'ah')")

# Load models
@st.cache_resource
def load_models():
    return {
        "scaler": joblib.load("scaler.pkl"),
        "best": joblib.load("best_model.pkl"),
        "svm": joblib.load("model_svm.pkl"),
        "rf": joblib.load("model_rf.pkl"),
    }

models = load_models()
N_MFCC = 20  # expected MFCCs from Oxford dataset

# Extract features
def extract_features(y, sr, n_mfcc=N_MFCC):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1), mfccs

# Audio input
method = st.radio("Choose input method", ["📁 Upload .wav file", "🎙️ Record from microphone"])

audio_bytes = None
sr = 16000
y = None
filename = None

if method == "📁 Upload .wav file":
    uploaded_file = st.file_uploader("Upload your .wav file", type=["wav"])
    if uploaded_file is not None:
        filename = uploaded_file.name
        y, sr = librosa.load(uploaded_file, sr=sr)
        audio_bytes = uploaded_file.read()

elif method == "🎙️ Record from microphone":
    if mic_available:
        audio_data = st_audiorec()
        if audio_data is not None:
            filename = "recorded_audio.wav"
            wav_io = BytesIO()
            sf.write(wav_io, audio_data, sr, format="WAV")
            wav_io.seek(0)
            audio_bytes = wav_io.read()
            y, sr = librosa.load(BytesIO(audio_bytes), sr=sr)
    else:
        st.warning("🎙️ Microphone recording not available. Please install `streamlit-audiorec`.")

if audio_bytes and y is not None:
    try:
        st.subheader("🎧 Playback")
        st.audio(audio_bytes, format='audio/wav')

        features_mean, mfcc_full = extract_features(y, sr)
        scaled = models["scaler"].transform([features_mean])

        # Display spectrogram
        st.subheader("📈 MFCC Spectrogram")
        fig, ax = plt.subplots()
        librosa.display.specshow(mfcc_full, x_axis='time', sr=sr, ax=ax)
        ax.set_title("MFCC")
        fig.colorbar(ax.images[0], ax=ax)
        st.pyplot(fig)

        # Predictions
        results = {}
        for name in ["best", "svm", "rf"]:
            prob = models[name].predict_proba(scaled)[0][1]
            pred = models[name].predict(scaled)[0]
            results[name] = {
                "prediction": "Positive" if pred == 1 else "Negative",
                "confidence": f"{prob*100:.2f}%"
            }

        st.subheader("🧪 Model Results")
        df_results = pd.DataFrame(results).T
        st.dataframe(df_results)

        # Radar chart
        fig_radar = go.Figure()
        for model in df_results.index:
            conf = float(df_results.loc[model]['confidence'].replace('%', ''))
            fig_radar.add_trace(go.Scatterpolar(
                r=[conf], theta=[model], fill='toself', name=model))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True, title="Confidence Radar Chart"
        )
        st.plotly_chart(fig_radar)

        # Final ensemble
        from collections import Counter
        final = Counter([v['prediction'] for v in results.values()]).most_common(1)[0][0]
        st.success(f"🎯 Final Prediction: **{final}**")

        # Log results
        log_row = {"file": filename or "recorded", **{f"{k}_prediction": v['prediction'] for k, v in results.items()}, **{f"{k}_conf": v['confidence'] for k, v in results.items()}}
        log_path = "prediction_log.csv"
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
            log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
        else:
            log_df = pd.DataFrame([log_row])
        log_df.to_csv(log_path, index=False)

        csv = df_results.to_csv(index=True).encode('utf-8')
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
