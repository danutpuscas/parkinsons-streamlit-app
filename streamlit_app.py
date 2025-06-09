import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
import plotly.graph_objects as go
import os
from collections import Counter
from io import BytesIO
import soundfile as sf

# Optional mic support
try:
    from streamlit_audiorec import st_audiorec
    mic_available = True
except ImportError:
    mic_available = False

st.set_page_config(page_title="Parkinson's Detection from Voice", layout="centered")
st.title("🧠 Parkinson's Detection from Voice")
st.write("Upload a .wav file or record a vowel sound (e.g., 'ah') for Parkinson's prediction")

@st.cache_resource
def load_models():
    return {
        'best': joblib.load('best_model.pkl'),
        'svm': joblib.load('model_svm.pkl'),
        'rf': joblib.load('model_rf.pkl'),
        'scaler': joblib.load('scaler.pkl')
    }

models = load_models()
st.markdown("### Input Source")
method = st.radio("Choose input method:", ["📁 Upload .wav file", "🎙️ Record from microphone"])

sr = 16000
n_mfcc = 24  # Must match what the model was trained on
y = None
filename = ""
audio_bytes = None

if method == "📁 Upload .wav file":
    uploaded_file = st.file_uploader("Upload your .wav file", type=["wav"])
    if uploaded_file is not None:
        filename = uploaded_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        y, sr = librosa.load(tmp_path, sr=sr)
        audio_bytes = open(tmp_path, 'rb').read()

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
        st.warning("🎙️ Microphone recording not available. Please install streamlit-audiorec")

if y is not None and audio_bytes is not None:
    st.audio(audio_bytes, format='audio/wav')

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1).reshape(1, -1)

    try:
        scaled = models['scaler'].transform(mfcc_mean)
        st.subheader("🎧 File Details")
        st.markdown(f"**Filename:** `{filename}`")
        st.markdown(f"**Duration:** `{librosa.get_duration(y=y, sr=sr):.2f} seconds`")

        st.subheader("📈 MFCC Spectrogram")
        fig, ax = plt.subplots()
        mfcc_img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax)
        fig.colorbar(mfcc_img, ax=ax, format='%+2.0f dB')
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax2)
        ax2.set_title("Waveform")
        st.pyplot(fig2)

        results = {}
        for name in ['best', 'svm', 'rf']:
            prob = models[name].predict_proba(scaled)[0][1]
            pred = models[name].predict(scaled)[0]
            results[name] = {'prediction': 'Positive' if pred == 1 else 'Negative', 'confidence': f"{prob*100:.2f}%"}

        st.subheader("🧪 Results")
        df_results = pd.DataFrame(results).T
        st.dataframe(df_results)

        fig_radar = go.Figure()
        for model in df_results.index:
            conf = float(df_results.loc[model]['confidence'].replace('%',''))
            fig_radar.add_trace(go.Scatterpolar(r=[conf], theta=[model], fill='toself', name=model))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="Confidence Radar Chart")
        st.plotly_chart(fig_radar)

        final_pred = Counter([v['prediction'] for v in results.values()]).most_common(1)[0][0]
        st.success(f"🎯 Final Ensemble Prediction: **{final_pred}**")

        csv = df_results.to_csv(index=True).encode('utf-8')
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")

        log_row = {"file": filename, **{f"{k}_prediction": v['prediction'] for k,v in results.items()}, **{f"{k}_conf": v['confidence'] for k,v in results.items()}}
        log_path = "predictions_log.csv"
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
            log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
        else:
            log_df = pd.DataFrame([log_row])
        log_df.to_csv(log_path, index=False)

    except Exception as e:
        st.error(f"Error processing file: {e}")
