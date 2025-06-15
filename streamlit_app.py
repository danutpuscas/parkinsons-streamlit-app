import streamlit as st
import numpy as np
import joblib
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import parselmouth
from parselmouth.praat import call
import tempfile
import plotly.graph_objects as go

st.set_page_config(page_title="Parkinson's Detection", layout="centered")
st.title("🧠 Parkinson's Detection (Voice Biomarkers)")
st.write("Upload a `.wav` file. The app extracts voice features and predicts Parkinson's disease.")

@st.cache_resource
def load_assets():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("threshold.txt", "r") as f:
        threshold = float(f.read().strip())
    return model, scaler, threshold

model, scaler, threshold = load_assets()
expected_feature_count = scaler.n_features_in_

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

def extract_features(audio_path):
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

    for feat in expected_features:
        if feat not in features or features[feat] is None:
            raise ValueError(f"Missing or invalid feature: {feat}")

    return features

file = st.file_uploader("Upload a `.wav` file", type=["wav"])

if file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.read())
            audio_path = tmp.name

        st.audio(audio_path, format='audio/wav')

        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        fig, ax = plt.subplots()
        librosa.display.specshow(mfccs, x_axis='time', ax=ax)
        ax.set(title='MFCC Visualization')
        fig.colorbar(ax.images[0], ax=ax)
        st.pyplot(fig)

        features = extract_features(audio_path)
        df = pd.DataFrame([features])
        st.subheader("📊 Extracted Features")
        st.write(df)

        if df.shape[1] != expected_feature_count:
            raise ValueError(f"Expected {expected_feature_count} features, got {df.shape[1]}.")
        if df.isnull().values.any():
            raise ValueError("Missing values in extracted features.")

        input_data = df.values
        scaled = scaler.transform(input_data)

        probas = model.predict_proba(scaled)
        probability = probas[:, 1][0]
        prediction = 1 if probability > threshold else 0

        st.subheader("🧪 Prediction Result")
        st.write(f"**Prediction**: {'🟥 Positive (Parkinson\'s)' if prediction == 1 else '🟩 Negative (Healthy)'}")
        st.write(f"**Confidence**: {probability * 100:.2f}% (threshold = {threshold})")

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[probability * 100],
            theta=['Confidence'],
            fill='toself',
            name='Model Confidence'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title="Confidence Radar"
        )
        st.plotly_chart(fig_radar)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
