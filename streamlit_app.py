import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Parkinson's Detector (CSV)", layout="centered")
st.title("🧠 Parkinson's Detection from Extracted Audio Features")
st.write("Upload a `.csv` file generated from Colab (20 features). The app will scale it and use a trained model to predict Parkinson's disease.")

# Load model, scaler, threshold
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("threshold.txt", "r") as f:
        threshold = float(f.read().strip())
    return model, scaler, threshold

model, scaler, threshold = load_artifacts()

# Upload CSV
uploaded_file = st.file_uploader("📄 Upload audio_features_full.csv", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Extracted Features")
        st.dataframe(df)

        # Check expected number of features
        if df.shape[1] != scaler.n_features_in_:
            raise ValueError(f"Expected {scaler.n_features_in_} features, got {df.shape[1]}.")

        if df.isnull().values.any():
            st.error("⚠️ CSV contains missing values. Please verify in Colab.")
        else:
    # Scale input
    scaled = scaler.transform(df)

    # Predict
    proba = model.predict_proba(scaled)[0][1]  # Probability of class 1 (Parkinson's)
    prediction = int(proba > threshold)

    st.subheader("🧪 Prediction Result")
    st.markdown(f"**Prediction:** {'🟥 Positive' if prediction == 1 else '🟩 Negative'}")
    st.markdown(f"**Confidence:** {proba * 100:.2f}%")
    st.markdown(f"**Threshold Used:** {threshold:.2f}")

    # Radar chart
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
        title="Confidence Radar"
    )
    st.plotly_chart(fig)
