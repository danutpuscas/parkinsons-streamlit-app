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
from streamlit_audiorec import st_audiorec
import soundfile as sf
from pydub import AudioSegment
from io import BytesIO

st.set_page_config(page_title="Parkinson's Detection from Voice", layout="centered")
st.title("🧠 Parkinson's Detection from Voice")
st.write("Upload a .wav file or record your voice (sustained vowel like 'ah')")

@st.cache_resource
def load_models():
    return {
        'best': joblib.load('best_model.pkl'),
        'svm': joblib.load('model_svm.pkl'),
        'rf': joblib.load('model_rf.pkl'),
        'scaler': joblib.load('scaler.pkl')
    }

models = load_models()

with open("feature_config.txt", "r") as f:
    n_mfcc = int(f.read().split("=")[1])

upload_method = st.radio("Choose input method:", ("Upload WAV file", "Record from microphone"))

audio_bytes = None
uploaded_file = None

if upload_method == "Upload WAV file":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
elif upload_method == "Record from microphone":
    st.info("Click to record your voice (ideal 3–5 seconds)")
    audio_bytes = st_audiorec()


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y, _ = librosa.effects.trim(y)  # Trim silence
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1), mfccs, sr, y

if uploaded_file or audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        if uploaded_file:
            tmp.write(uploaded_file.read())
        else:
            # Convert bytes to wav
            audio = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")
            audio.export(tmp.name, format="wav")
        tmp_path = tmp.name

    try:
        features_mean, mfcc_full, sr, y = extract_features(tmp_path)
        scaled = models['scaler'].transform([features_mean])

        st.subheader("🎧 File Details")
        st.markdown(f"**Duration:** `{librosa.get_duration(y=y, sr=sr):.2f} seconds`")
        st.audio(tmp_path, format="audio/wav")

        st.subheader("📈 MFCC Spectrogram")
        fig, ax = plt.subplots()
        mfcc_img = librosa.display.specshow(mfcc_full, x_axis='time', ax=ax, sr=sr)
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

        # Radar chart
        fig_radar = go.Figure()
        for model in df_results.index:
            conf = float(df_results.loc[model]['confidence'].replace('%',''))
            fig_radar.add_trace(go.Scatterpolar(r=[conf], theta=[model], fill='toself', name=model))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="Confidence Radar Chart")
        st.plotly_chart(fig_radar)

        # Ensemble decision
        from collections import Counter
        final_pred = Counter([v['prediction'] for v in results.values()]).most_common(1)[0][0]
        st.success(f"🎯 Final Ensemble Prediction: **{final_pred}**")

        csv = df_results.to_csv(index=True).encode('utf-8')
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")

        # Append to log
        log_row = {"file": uploaded_file.name if uploaded_file else "mic_recording", **{f"{k}_prediction": v['prediction'] for k,v in results.items()}, **{f"{k}_conf": v['confidence'] for k,v in results.items()}}
        log_path = "predictions_log.csv"
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
            log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
        else:
            log_df = pd.DataFrame([log_row])
        log_df.to_csv(log_path, index=False)

    except Exception as e:
        st.error(f"Error processing file: {e}")
