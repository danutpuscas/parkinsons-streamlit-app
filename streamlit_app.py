import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import tempfile
import pandas as pd

# Must be first Streamlit command
st.set_page_config(page_title="Parkinson's Detection from Voice", layout="centered")
st.title("🧠 Parkinson's Detection from Voice")
st.write("Upload a .wav file of a sustained vowel sound (e.g., 'ah')")

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
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=19)
    return np.mean(mfccs, axis=1), mfccs, sr

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        features_mean, mfcc_full, sr = extract_features(tmp_path)
        scaled = models['scaler'].transform([features_mean])

        st.subheader("📈 MFCC Spectrogram")
        fig, ax = plt.subplots()
        librosa.display.specshow(mfcc_full, x_axis='time', ax=ax, sr=sr)
        plt.colorbar(format='%+2.0f dB')
        st.pyplot(fig)

        results = {}
        for name in ['best', 'svm', 'rf']:
            prob = models[name].predict_proba(scaled)[0][1]
            pred = models[name].predict(scaled)[0]
            results[name] = {'prediction': 'Positive' if pred == 1 else 'Negative', 'confidence': f"{prob*100:.2f}%"}

        st.subheader("🧪 Results")
        df_results = pd.DataFrame(results).T
        st.dataframe(df_results)

        csv = df_results.to_csv(index=True).encode('utf-8')
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
