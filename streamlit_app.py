import streamlit as st
st.set_page_config(page_title="Parkinson's Voice Detection", layout="centered")

# Workaround for torch class watcher issue
import sys
if "__path__" in sys.modules:
    del sys.modules["__path__"]

import numpy as np
import torch
import joblib
import librosa
import os
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🎙 Parkinson's Disease Detection from Voice")
st.markdown("Upload a `.wav` voice recording and detect the presence of Parkinson's symptoms using a pre-trained ML model.")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

def extract_features(file_path, n_mfcc=19):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        st.audio("temp.wav")
        features = extract_features("temp.wav")

        # Reshape and scale
        features_reshaped = features.reshape(1, -1)
        if features_reshaped.shape[1] != scaler.mean_.shape[0]:
            st.error(f"Expected {scaler.mean_.shape[0]} features, but got {features_reshaped.shape[1]}")
        else:
            scaled_features = scaler.transform(features_reshaped)
            prediction = model.predict(scaled_features)[0]
            prob = model.predict_proba(scaled_features)[0][1] if hasattr(model, "predict_proba") else None

            if prediction == 1:
                st.error("⚠️ Parkinson's symptoms detected.")
            else:
                st.success("✅ No Parkinson's symptoms detected.")

            if prob is not None:
                st.write(f"Probability of Parkinson's: **{prob:.2%}**")
    except Exception as e:
        st.error(f"Error processing file: {e}")
