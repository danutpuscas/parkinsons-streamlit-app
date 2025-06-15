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
st.write("Upload a `.wav` file. The app extracts biomedical voice features (jitter, shimmer, HNR, etc.) and predicts Parkinson's disease using a pre-trained model.")

# Load model and scaler
@st.cache_resource
def load_models():
    return {
        'best': joblib.load('best_model.pkl'),
        'scaler': joblib.load('scaler.pkl')
    }

models = load_models()
expected_feature_count = models['scaler'].n_features_in_

# Feature list from model training
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

# Feature extraction
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

    for feat in expected_features:
        if feat not in features or features[feat] is None:
            raise ValueError(f"Missing or invalid feature: {feat}")

    return features

# Upload interface
file = st.file_uploader("Upload a `.wav` file", type=["wav"])

if file is not None:
    try:
        # Save temp .wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.read())
            audio_path = tmp.name

        st.audio(audio_path, format='audio/wav')

        # Optional: Visualize MFCCs
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
        st.subheader("📊 Extracted Features")
        st.write(df)

        # Check dimensions
        if df.shape[1] != expected_feature_count:
            raise ValueError(f"Expected {expected_feature_count} features, got {df.shape[1]}.")
        if df.isnull().values.any():
            raise ValueError("Missing values in extracted features.")

        # Scale
        input_data = df.values
        scaled = models['scaler'].transform(input_data)

        if scaled.shape[0] == 0:
            raise ValueError("Empty input after scaling.")

        # Predict (with fallback)
        try:
            preds = models['best'].predict(scaled)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

        try:
            probs = models['best'].predict_proba(scaled)[:, 1]
            avg_conf = float(np.mean(probs))
        except AttributeError:
            avg_conf = 1.0  # fallback when model doesn't support predict_proba

        result = int(np.round(np.mean(preds)))

        # Display
        st.subheader("🧪 Prediction Result")
        st.write(f"**Prediction**: {'🟥 Positive' if result == 1 else '🟩 Negative'}")
        if avg_conf <= 1.0:
            st.write(f"**Confidence**: {avg_conf * 100:.2f}%")
        else:
            st.write("**Confidence**: N/A (model does not support probabilities)")

        # Radar plot
        if avg_conf <= 1.0:
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
                title="Confidence Radar"
            )
            st.plotly_chart(fig_radar)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
