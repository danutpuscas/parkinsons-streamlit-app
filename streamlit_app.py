import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
from collections import Counter

st.set_page_config(page_title="Parkinson's Detection from MFCC", layout="centered")
st.title("🧠 Parkinson's Detection from MFCC Features")
st.write("Upload a .wav file or a .csv/.xlsx file with MFCCs to predict Parkinson's.")

@st.cache_resource
def load_resources():
    best_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return best_model, scaler

model, scaler = load_resources()
num_features = scaler.n_features_in_  # Automatically infer feature count

file = st.file_uploader("Upload MFCC file (.csv, .xlsx)", type=["csv", "xlsx"])

if file is not None:
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format")

        mfcc_data = df.iloc[:, :num_features].values
        scaled = scaler.transform(mfcc_data)

        # Model prediction
        probs = model.predict_proba(scaled)[:, 1]
        preds = model.predict(scaled)
        majority_vote = int(np.round(np.mean(preds)))
        avg_conf = np.mean(probs)

        result = {
            'prediction': 'Positive' if majority_vote == 1 else 'Negative',
            'confidence': f"{avg_conf * 100:.2f}%"
        }

        st.subheader("🧪 Results")
        df_results = pd.DataFrame([result], index=["Best Model"])
        st.dataframe(df_results)

        # Radar chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[avg_conf * 100],
            theta=["Best Model"],
            fill='toself',
            name="Best Model"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Confidence Radar Chart"
        )
        st.plotly_chart(fig_radar)

        st.success(f"🎯 Final Prediction: **{result['prediction']}**")

        csv = df_results.to_csv(index=True).encode('utf-8')
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
