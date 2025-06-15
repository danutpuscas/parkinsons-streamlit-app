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

# Page setup
st.set_page_config(page_title="Parkinson's Detection", layout="centered")
st.title("🧠 Parkinson's Detection (Voice Analysis)")
st.write("Upload a `.wav` file to extract biomedical voice features and predict Parkinson's.")

# Load model and scaler
@st.cache_resource
def load_models():
    return {
        'best': joblib.load('best_model.pkl'),
        'scaler': joblib.load('scaler.pkl')
    }

models = load_models()

# Show expected input features
expected_feature_count = models['scaler'].n_features_in_
st.write(f"🔧 Model expects **{expected_feature_count} features**.")

# List of features the model was trained on
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

# Extraction function
def extract_biomedical_features(audio_path):
    try:
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

        # Ensure all expected features exist
        for feat in expected_features:
            if feat not in features or features[feat] is None:
                raise ValueError(f"Missing or invalid feature: {feat}")
        return features
    except Exception as e:
        raise RuntimeError(f"Error extracting features: {e}")

# File upload
file = st.file_uploader("Upload a `.wav` file", type=["wav"])

if file is not None:
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.read())
            audio_path = tmp.name

        st.audio(audio_path, format='audio/wav')

        # Optional MFCC visual
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        fig, ax = plt.subplots()
        librosa.display.specshow(mfccs, x_axis='time', ax=ax)
        ax.set(title='MFCC Visualization')
        fig.colorbar(ax.images[0], ax=ax)
        st.pyplot(fig)

        # Feature extraction
        st.subheader("📊 Extracting Biomedical Features...")
        features = extract_biomedical_features(audio_path)
        df = pd.DataFrame([features])
        st.write("✅ Extracted Features:")
        st.write(df)

        # Sanity checks
        if df.isnull().values.any():
            raise ValueError("NaN values found in extracted features.")
        if df.shape[1] != expected_feature_count:
            raise ValueError(f"Feature mismatch: Expected {expected_feature_count}, got {df.shape[1]}.")

        # Scale and predict
        input_data = df.values
        scaled = models['scaler'].transform(input_data)
        if scaled.shape[0] == 0:
            raise ValueError("No data passed to model.")

        probs = models['best'].predict_proba(scaled)[:, 1]
        preds = models['best'].predict(scaled)
        result = int(np.round(np.mean(preds)))
        avg_conf = float(np.mean(probs))

        st.subheader("🧪 Prediction Result")
        st.write(f"**Prediction**: {'Positive' if result == 1 else 'Negative'}")
        st.write(f"**Confidence**: {avg_conf * 100:.2f}%")

        # Radar chart
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
            title="Model Confidence Radar"
        )
        st.plotly_chart(fig_radar)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
