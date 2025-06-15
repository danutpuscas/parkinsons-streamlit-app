import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import librosa
import librosa.display
import matplotlib.pyplot as plt
import parselmouth
from parselmouth.praat import call
import tempfile

st.set_page_config(page_title="Parkinson's Detection", layout="centered")
st.title("🧠 Parkinson's Detection from Voice Features")
st.write("Upload a .wav file or .csv/.xlsx file to predict Parkinson's disease using biomedical voice features.")

@st.cache_resource
def load_models():
    return {
        'best': joblib.load('best_model.pkl'),
        'scaler': joblib.load('scaler.pkl')
    }

models = load_models()

def extract_biomedical_features(audio_path):
    snd = parselmouth.Sound(audio_path)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)

    features = {
        'Jitter (%)': call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
        'Jitter (Abs)': call(point_process, "Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3),
        'Shimmer (dB)': call([snd, point_process], "Get shimmer (dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        'Shimmer (APQ3)': call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        'Shimmer (APQ5)': call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        'Shimmer (APQ11)': call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        'Shimmer (DDA)': call([snd, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        'HNR': call(harmonicity, "Get mean", 0, 0),
    }
    return features

file = st.file_uploader("Upload a voice file or feature table (.wav, .csv, .xlsx)", type=["csv", "xlsx", "wav"])

if file is not None:
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
            input_data = df.values

        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
            input_data = df.values

        elif file.name.endswith(".wav"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file.read())
                audio_path = tmp.name

            st.audio(audio_path, format='audio/wav')

            # Show MFCC (optional visualization)
            y, sr = librosa.load(audio_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            fig, ax = plt.subplots()
            librosa.display.specshow(mfccs, x_axis='time', ax=ax)
            ax.set(title='MFCC Visualization')
            fig.colorbar(ax.images[0], ax=ax)
            st.pyplot(fig)

            # Extract biomedical voice features
            features = extract_biomedical_features(audio_path)
            df = pd.DataFrame([features])
            st.subheader("📊 Extracted Features")
            st.write(df)

            input_data = df.values

        else:
            raise ValueError("Unsupported file format.")

        # Scale and predict
        scaled = models['scaler'].transform(input_data)
        probs = models['best'].predict_proba(scaled)[:, 1]
        preds = models['best'].predict(scaled)
        majority_vote = int(np.round(np.mean(preds)))
        avg_conf = np.mean(probs)

        st.subheader("🧪 Prediction Result")
        result = "Positive" if majority_vote == 1 else "Negative"
        st.write(f"**Prediction**: {result}")
        st.write(f"**Confidence**: {avg_conf*100:.2f}%")

        # Radar Chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=[avg_conf * 100], theta=['Confidence'], fill='toself', name='Model'))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Confidence Radar Chart"
        )
        st.plotly_chart(fig_radar)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
