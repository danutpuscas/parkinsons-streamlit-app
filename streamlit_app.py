import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import io
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Set page config
st.set_page_config(page_title="Parkinson's Voice Detection", layout="wide")

st.title("🧠 Parkinson's Voice Detection App")
st.write("Upload a `.wav` file to predict the likelihood of Parkinson's disease.")

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Additional models for comparison
model_svm = joblib.load("model_svm.pkl")
model_rf = joblib.load("model_rf.pkl")

# Function to extract MFCCs
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean, mfcc, sr

# Function to convert plot to image bytes
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf.read()

# Handle file upload
uploaded_file = st.file_uploader("Upload your .wav file here", type=["wav"])

if uploaded_file is not None:
    try:
        # Save to a temporary file
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract features
        features, mfcc_matrix, sr = extract_features("temp.wav")
        features_reshaped = features.reshape(1, -1)

        expected_features = scaler.n_features_in_
        if features_reshaped.shape[1] != expected_features:
            st.error(f"Expected {expected_features} features, but got {features_reshaped.shape[1]}")
        else:
            # Scale features
            scaled = scaler.transform(features_reshaped)

            # Predictions from multiple models
            main_pred = model.predict(scaled)[0]
            main_prob = model.predict_proba(scaled)[0][1] if hasattr(model, "predict_proba") else None

            svm_prob = model_svm.predict_proba(scaled)[0][1] if hasattr(model_svm, "predict_proba") else 0
            rf_prob = model_rf.predict_proba(scaled)[0][1] if hasattr(model_rf, "predict_proba") else 0

            # Display results
            st.subheader("Prediction Result")
            if main_pred == 1:
                st.error("⚠️ Parkinson's symptoms detected.")
            else:
                st.success("✅ No Parkinson's symptoms detected.")

            st.metric("Model Confidence", f"{main_prob:.2%}")

            # Show comparison with other models
            st.subheader("Model Confidence Comparison")
            comparison_df = pd.DataFrame({
                "Model": ["Main (Best)", "SVM", "Random Forest"],
                "Probability": [main_prob, svm_prob, rf_prob]
            })
            st.bar_chart(comparison_df.set_index("Model"))

            # Show MFCC spectrogram
            st.subheader("MFCC Spectrogram")
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.specshow(mfcc_matrix, x_axis='time', ax=ax)
            ax.set_title('MFCC')
            plt.colorbar(format='%+2.0f dB')
            st.pyplot(fig)

            # Downloadable report
            st.subheader("Download Prediction Report")
            report = pd.DataFrame({
                "Model": ["Main (Best)", "SVM", "Random Forest"],
                "Predicted_Probability": [main_prob, svm_prob, rf_prob]
            })
            csv = report.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="prediction_report.csv">Download CSV Report</a>'
            st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")

# Add footer
st.markdown("---")
st.caption("Developed for academic research on Parkinson's disease detection using voice analysis.")
