import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# UI Configuration
st.set_page_config(page_title="Parkinson's Detector", layout="centered")
st.title("🧠 Parkinson's Detection from Audio Features")
st.write("Upload a `.csv` file extracted from a `.wav` recording.")

# Load trained model, scaler, and threshold
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("threshold.txt", "r") as f:
        threshold = float(f.read().strip())
    return model, scaler, threshold

model, scaler, threshold = load_artifacts()

# Expected feature order
expected_columns = [
    'Jitter(%)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
    'Shimmer', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
    'HNR', 'RPDE', 'DFA', 'PPE'
]

# Upload CSV
uploaded_file = st.file_uploader("📄 Upload extracted_features.csv", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Extracted Features (Preview)")
        st.dataframe(df)

        # Subset to only the expected columns
        try:
            df = df[expected_columns]
        except KeyError as e:
            st.error(f"❌ Uploaded CSV is missing one or more required features: {e}")
            st.stop()

        # Handle missing values
        if df.isnull().values.any():
            st.warning("⚠️ Missing values found in the file. Imputing with column means...")
            df.fillna(df.mean(), inplace=True)

        # Scale features
        scaled = scaler.transform(df)

        # Predict
        proba = model.predict_proba(scaled)[0][1]  # Class 1 = Parkinson's
        prediction = int(proba > threshold)

        st.subheader("🧪 Prediction Result")
        st.markdown(f"**Prediction:** {'🟥 Parkinson Detected' if prediction == 1 else '🟩 Healthy'}")

        # Radar plot
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

    except Exception as e:
        st.error(f"❌ Failed to process the file: {e}")
