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
st.write("Upload a `.wav`, `.csv`, or `.xlsx` file to predict Parkinson's disease using biomedical voice features.")

@st.cache_resource
def load_models():
    return {
        'best': joblib.load('best_model.pkl'),
        'scaler': joblib.load('scaler.pkl')
    }

models = load_models()

expected_features = [
    'Jitter (%)',
    'Jitter (Abs)',
    'Shimmer (dB)',
    'Shimmer (APQ3)',
    'Shimmer (APQ5)',
    'Shimmer (APQ11)',
    'Shimmer (DDA)',
    'HNR'
]

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
        'HNR': call(harmonicity, "Get mean", 0, 0)
    }

    # Validate features
    for feat in expected_features:
        if feat not in features:
            raise ValueError(f"Missing feature: {feat}")

    return features

file = st.file_uploader("Upload file", type=["csv", "xlsx", "wav"])

if file is not None:
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)

        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)

        elif file.name.endswith(".wav"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file.read())
                audio_path = tmp.name

            st.audio(audio_path, format='audio/wav')

            # Optional: Display MFCC plot
            y, sr = librosa.load(audio_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            fig, ax = plt.subplots()
            librosa.display.specshow(mfccs, x_axis='time', ax=ax)
            ax.set(title='MFCC Visualization')
            fig.colorbar(ax.images[0], ax=ax)
            st.pyplot(fig)

            # Extract features
            features = extract_biomedical_features(audio_path)
            df = pd.DataFrame([features])

        else:
            raise ValueError("Unsupported file format.")

        st.subheader("📊 Extracted Features")
        st.write(df)

        if df.isnull().values.any():
            raise ValueError("Missing values found in extracted features.")

        if df.shape[1] != models['scaler'].n_features_in_:
            raise ValueError(f"Feature mismatch: expected {models['scaler'].n_features_in_}, got {df.shape[1]}")

        input_data = df.values
        scaled = models['scaler'].transform(input_data)

        # Prediction
        probs = models['best'].predict_proba(scaled)[:, 1]
        preds = models['best'].predict(scaled)
        majority_vote = int(np.round(np.mean(preds)))
        avg_conf = np.mean(probs)

        st.subheader("🧪 Prediction Result")
        result = "Positive" if majority_vote == 1 else "Negative"
        st.write(f"**Prediction**: {result}")
        st.write(f"**Confidence**: {avg_conf * 100:.2f}%")

        # Radar plot
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[avg_conf * 100],
            theta=['Confidence'],
            fill='toself',
            name='Model Confidence'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title="Confidence Radar Chart"
        )
        st.plotly_chart(fig_radar)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
