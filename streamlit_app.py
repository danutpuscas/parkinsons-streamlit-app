import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import tempfile

st.set_page_config(page_title="Parkinson's Detection from MFCC", layout="centered")
st.title("🧠 Parkinson's Detection from MFCC Features")
st.write("Upload a .wav file or a .csv/.xlsx file with MFCCs to predict Parkinson's.")

@st.cache_resource
def load_models():
    return {
        'best': joblib.load('best_model.pkl'),
        'scaler': joblib.load('scaler.pkl')
    }

models = load_models()

# Define expected number of MFCC features (you can change this if needed)
NUM_FEATURES = 20

file = st.file_uploader("Upload MFCC file or Audio (.csv, .xlsx, .wav)", type=["csv", "xlsx", "wav"])

if file is not None:
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
            mfcc_data = df.iloc[:, :NUM_FEATURES].values

        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
            mfcc_data = df.iloc[:, :NUM_FEATURES].values

        elif file.name.endswith(".wav"):
            # Save temporarily to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(file.read())
                audio_path = tmp_file.name

            st.audio(audio_path, format='audio/wav')

            # Load and extract MFCCs
            y, sr = librosa.load(audio_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_FEATURES)
            mfcc_data = mfccs.T

            # Display MFCC spectrogram
            fig, ax = plt.subplots()
            img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
            ax.set(title='MFCC')
            fig.colorbar(img, ax=ax)
            st.pyplot(fig)

        else:
            raise ValueError("Unsupported file format")

        scaled = models['scaler'].transform(mfcc_data)
        probs = models['best'].predict_proba(scaled)[:, 1]
        preds = models['best'].predict(scaled)
        majority_vote = int(np.round(np.mean(preds)))
        avg_conf = np.mean(probs)

        st.subheader("🧪 Result")
        result_text = "Positive" if majority_vote == 1 else "Negative"
        st.write(f"**Prediction**: {result_text}")
        st.write(f"**Confidence**: {avg_conf*100:.2f}%")

        # Radar plot
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=[avg_conf * 100], theta=['Confidence'], fill='toself', name='Best Model'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="Confidence Radar Chart")
        st.plotly_chart(fig_radar)

    except Exception as e:
        st.error(f"Error processing file: {e}")
