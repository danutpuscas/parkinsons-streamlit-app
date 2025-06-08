import streamlit as st
import numpy as np
import torch
import joblib
import librosa
import os
import tempfile
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# SECTION 1: Load model and scaler
# -------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('best_model.pkl')  # e.g. RandomForest, SVM etc.
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# -------------------------------
# SECTION 2: Feature extraction from .wav
# -------------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=19)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean.reshape(1, -1)

# -------------------------------
# SECTION 3: Streamlit UI
# -------------------------------
st.set_page_config(page_title="Parkinson’s Detection from Voice", layout="centered")
st.title("🧠 Parkinson’s Voice Detection")
st.write("Upload a `.wav` file to check for Parkinson's-related vocal biomarkers.")

uploaded_file = st.file_uploader("Upload a voice recording (.wav)", type=["wav"])

# -------------------------------
# SECTION 4: Inference
# -------------------------------
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    try:
        features = extract_features(file_path)

        if features.shape[1] != scaler.mean_.shape[0]:
            st.error(f"Feature mismatch: Model expects {scaler.mean_.shape[0]} features, but got {features.shape[1]}")
        else:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            prob = getattr(model, "predict_proba", lambda x: [[0, 1]])(features_scaled)[0][1]

            st.subheader("Prediction Result:")
            st.success("✅ Parkinson’s Likely" if prediction == 1 else "🟢 No Parkinson’s Detected")
            st.metric(label="Confidence", value=f"{prob * 100:.2f}%")
    except Exception as e:
        st.error(f"Error during processing: {e}")
