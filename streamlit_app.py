import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page setup
st.set_page_config(page_title="Parkinson's Detection (CSV)", layout="centered")
st.title("🧠 Parkinson's Detection from Extracted Audio Features")
st.markdown("""
Upload a `.csv` file exported from your Colab extractor containing 20 audio features.
The app will scale it, handle any missing values, and use your trained model to predict Parkinson's presence.
""")

# Load model artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("threshold.txt", "r") as f:
        threshold = float(f.read().strip())
    return model, scaler, threshold

model, scaler, threshold = load_artifacts()

# Upload CSV
uploaded_file = st.file_uploader("📄 Upload extracted audio_features CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Raw Features")
        st.dataframe(df)

        # Handle missing values
        if df.isnull().values.any():
            st.warning("⚠️ Missing values found in the file. Imputing with column means...")
            df.fillna(df.mean(), inplace=True)

        # Check for proper shape
        if df.shape[1] != scaler.n_features_in_:
            st.error(f"❌ Expected {scaler.n_features_in_} features, got {df.shape[1]}. Please verify the CSV.")
        else:
            # Scale and predict
            scaled = scaler.transform(df)
            proba = model.predict_proba(scaled)[0][1]
            prediction = int(proba > threshold)

            st.subheader("🧪 Prediction Result")
            st.markdown(f"**Prediction:** {'🟥 Parkinson\'s Detected' if prediction == 1 else '🟩 Healthy (No Parkinson\'s)'}")
            st.markdown(f"**Confidence:** `{proba * 100:.2f}%`")
            st.markdown(f"**Threshold Used:** `{threshold:.2f}`")

            # Radar chart (visual confidence)
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[proba * 100],
                theta=['Model Confidence'],
                fill='toself',
                name='Confidence'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title="🧭 Confidence Radar"
            )
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"❌ Failed to process CSV: {e}")
