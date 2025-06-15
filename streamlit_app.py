import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

st.set_page_config(page_title="🧠 Parkinson's Voice Detection", layout="centered")

st.title("🧬 Parkinson's Prediction from Voice Biomarkers")
st.markdown("Upload the extracted `audio_features.csv` and optional `voice_diagnostic.txt`.")

# Load model, scaler, threshold
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        with open("threshold.txt", "r") as f:
            threshold = float(f.read().strip())
        return model, scaler, threshold
    except Exception as e:
        st.error(f"❌ Failed to load model assets: {e}")
        return None, None, None

model, scaler, threshold = load_assets()

# Upload files
csv_file = st.file_uploader("📄 Upload `audio_features.csv`", type="csv")
txt_file = st.file_uploader("📝 Upload `voice_diagnostic.txt` (optional)", type="txt")

# Process CSV
if csv_file and model and scaler:
    try:
        df = pd.read_csv(csv_file)
        st.subheader("📊 Extracted Voice Features")
        st.dataframe(df.T.rename(columns={0: "Value"}))

        # Radar chart (only non-null numeric columns)
        df_clean = df.dropna(axis=1)
        if not df_clean.empty:
            labels = df_clean.columns.tolist()
            values = df_clean.loc[0].tolist()
            values += values[:1]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels + [labels[0]],
                fill='toself'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title="Radar Chart of Voice Biomarkers",
                showlegend=False
            )
            st.plotly_chart(fig)
        else:
            st.warning("Not enough valid features for radar chart.")

        # Prediction
        features_scaled = scaler.transform(df.fillna(0))  # Fill missing with 0 for simplicity
        prob = model.predict_proba(features_scaled)[0][1]

        st.subheader("🧪 Parkinson’s Prediction")
        if prob > threshold:
            st.error(f"🔴 Prediction: High likelihood of Parkinson's (probability = {prob:.2f})")
        else:
            st.success(f"🟢 Prediction: No Parkinson's detected (probability = {prob:.2f})")

    except Exception as e:
        st.error(f"❌ Failed to process audio_features.csv: {e}")

# Optional Feedback
if txt_file:
    try:
        txt_content = txt_file.read().decode("utf-8")
        st.subheader("📝 Feedback from Colab")
        st.text(txt_content)
    except Exception as e:
        st.error(f"❌ Failed to read diagnostic file: {e}")
