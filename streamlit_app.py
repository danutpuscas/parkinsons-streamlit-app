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
st.write("Upload a `.wav` file of a sustained vowel sound (e.g., 'ah') to analyze Parkinson's likelihood.")

@st.cache_resource
def load_models():
    return {
        'best': joblib.load('best_model.pkl'),
        'svm': joblib.load('model_svm.pkl'),
        'rf': joblib.load('model_rf.pkl'),
        'scaler': joblib.load('scaler.pkl')
    }

models = load_models()
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mean_mfcc = np.mean(mfccs, axis=1)
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0

    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
    rmse = np.mean(librosa.feature.rms(y=y))
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return np.concatenate([mean_mfcc, [pitch_mean, zcr, rmse, cent]]), mfccs, sr, y

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        features, mfcc_full, sr, y = extract_features(tmp_path)
        scaled = models['scaler'].transform([features])

        st.audio(uploaded_file, format='audio/wav')

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

        st.subheader("🧪 Model Predictions")
        df_results = pd.DataFrame(results).T
        st.dataframe(df_results)

        fig_radar = go.Figure()
        for model in df_results.index:
            conf = float(df_results.loc[model]['confidence'].replace('%',''))
            fig_radar.add_trace(go.Scatterpolar(r=[conf], theta=[model], fill='toself', name=model))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="Confidence Radar Chart")
        st.plotly_chart(fig_radar)

        from collections import Counter
        final_pred = Counter([v['prediction'] for v in results.values()]).most_common(1)[0][0]
        st.success(f"🎯 Final Ensemble Prediction: **{final_pred}**")

        csv = df_results.to_csv(index=True).encode('utf-8')
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
