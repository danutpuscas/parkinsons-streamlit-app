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
st.write("Upload a .wav file or record live from your microphone")

@st.cache_resource
def load_models():
    return {
        'best': joblib.load('best_model.pkl'),
        'svm': joblib.load('model_svm.pkl'),
        'rf': joblib.load('model_rf.pkl'),
        'scaler': joblib.load('scaler.pkl')
    }

models = load_models()

# Read saved MFCC feature count from config
:contentReference[oaicite:1]{index=1}
    :contentReference[oaicite:2]{index=2}

# Two options: file upload or live record
:contentReference[oaicite:3]{index=3}

audio_bytes = None
:contentReference[oaicite:4]{index=4}
    :contentReference[oaicite:5]{index=5}
    if uploaded:
        :contentReference[oaicite:6]{index=6}

:contentReference[oaicite:7]{index=7}
    :contentReference[oaicite:8]{index=8}
        :contentReference[oaicite:9]{index=9}
        if rec:
            :contentReference[oaicite:10]{index=10}

if audio_bytes:
    # Do processing
    try:
        :contentReference[oaicite:11]{index=11}
        :contentReference[oaicite:12]{index=12}
            :contentReference[oaicite:13]{index=13}
            :contentReference[oaicite:14]{index=14}

        # Extract features
        y, sr = librosa.load(tmp_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        features_mean = np.mean(mfccs, axis=1)
        scaled = models['scaler'].transform([features_mean])

        # Show waveform and spectrogram
        st.subheader("🎧 Audio Visualizations")
        fig_wf, ax_wf = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax_wf)
        ax_wf.set_title("Waveform")
        st.pyplot(fig_wf)

        fig_sp, ax_sp = plt.subplots()
        img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax_sp)
        fig_sp.colorbar(img, ax=ax_sp)
        ax_sp.set_title("MFCC Spectrogram")
        st.pyplot(fig_sp)

        # Model predictions
        results = {}
        for name in ['best','svm','rf']:
            prob = models[name].predict_proba(scaled)[0][1]
            pred = models[name].predict(scaled)[0]
            results[name] = {
                'prediction': 'Positive' if pred==1 else 'Negative',
                'confidence': f"{prob*100:.1f}%"
            }

        st.subheader("🧪 Results")
        df = pd.DataFrame(results).T
        st.dataframe(df)

        # Confidence Radar
        fig_r = go.Figure()
        for idx,row in df.iterrows():
            fig_r.add_trace(go.Scatterpolar(r=[float(row.confidence[:-1])], theta=[idx], fill="toself", name=idx))
        fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,100])), showlegend=True)
        st.plotly_chart(fig_r)

        # Ensemble majority vote
        from collections import Counter
        final = Counter(df['prediction']).most_common(1)[0][0]
        st.success(f"🎯 Ensemble says: **{final}**")

        # Download results
        tocsv = df.to_csv().encode('utf-8')
        st.download_button("Download Results CSV", tocsv, "results.csv")

        # Save logfile
        row = {"file_mode": mode, **{f"{k}_pred":v['prediction'] for k,v in results.items()},
               **{f"{k}_conf":v['confidence'] for k,v in results.items()}}
        logf = "predictions_log.csv"
        logdf = pd.read_csv(logf) if os.path.exists(logf) else pd.DataFrame()
        logdf = pd.concat([logdf, pd.DataFrame([row])], ignore_index=True)
        logdf.to_csv(logf,index=False)

    except Exception as e:
        st.error(f"Error: {e}")
