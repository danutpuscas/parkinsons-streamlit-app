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

st.set_page_config(page_title="Parkinson's Detection from Voice", layout="centered")
st.title("🧠 Parkinson's Detection from Voice")
st.write("Upload or record a .wav file of a sustained vowel sound (e.g., 'ah')")

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

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features = np.mean(mfccs, axis=1)
    return features, mfccs, sr, y

record_mode = st.toggle("🎙️ Record from Microphone")
if record_mode:
    st.info("Please install streamlit-audiorec manually if not yet available.")
    try:
        from streamlit_audiorec import st_audiorec
        wav_audio_data = st_audiorec()
        if wav_audio_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                f.write(wav_audio_data)
                tmp_path = f.name
    except:
        st.warning("streamlit-audiorec not installed or not working. Please upload instead.")
        tmp_path = None
else:
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            f.write(uploaded_file.read())
            tmp_path = f.name
    else:
        tmp_path = None

if tmp_path:
    try:
        features, mfcc_full, sr, y = extract_features(tmp_path)
        scaled = models['scaler'].transform([features])

        st.subheader("🎧 Audio Preview")
        st.audio(tmp_path)
        st.markdown(f"**Duration:** `{librosa.get_duration(y=y, sr=sr):.2f} seconds`")

        st.subheader("📈 MFCC Spectrogram")
        fig, ax = plt.subplots()
        img = librosa.display.specshow(mfcc_full, x_axis='time', sr=sr, ax=ax)
        fig.colorbar(img, ax=ax)
        st.pyplot(fig)

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
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
        st.plotly_chart(fig_radar)

        from collections import Counter
        final_pred = Counter([v['prediction'] for v in results.values()]).most_common(1)[0][0]
        st.success(f"🎯 Final Ensemble Prediction: **{final_pred}**")

    except Exception as e:
        st.error(f"Error processing file: {e}")
