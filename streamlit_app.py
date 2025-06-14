import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Parkinson's Detection from MFCC", layout="centered")
st.title("🧠 Parkinson's Detection from MFCC Features")
st.write("Upload a .wav file or a .csv/.xlsx file with MFCCs to predict Parkinson's.")

@st.cache_resource
def load_models():
    return {
        'best': joblib.load('best_model.pkl'),
        'svm': joblib.load('model_svm.pkl'),
        'rf': joblib.load('model_rf.pkl'),
        'scaler': joblib.load('scaler.pkl')
    }

models = load_models()

with open("feature_config.txt", "r") as f:
    num_features = int(f.read())

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
        scaled = models['scaler'].transform(mfcc_data)

        results = {}
        for name in ['best', 'svm', 'rf']:
            probs = models[name].predict_proba(scaled)[:, 1]
            preds = models[name].predict(scaled)
            majority_vote = int(np.round(np.mean(preds)))
            avg_conf = np.mean(probs)

            results[name] = {
                'prediction': 'Positive' if majority_vote == 1 else 'Negative',
                'confidence': f"{avg_conf*100:.2f}%"
            }

        st.subheader("🧪 Results")
        df_results = pd.DataFrame(results).T
        st.dataframe(df_results)

        fig_radar = go.Figure()
        for model in df_results.index:
            conf = float(df_results.loc[model]['confidence'].replace('%',''))
            fig_radar.add_trace(go.Scatterpolar(r=[conf], theta=[model], fill='toself', name=model))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="Confidence Radar Chart")
        st.plotly_chart(fig_radar)

        from collections import Counter
        final_pred = Counter([v['prediction'] for v in results.values()]).most_common(1)[0][0]
        st.success(f"🎯 Final Ensemble Prediction: **{final_pred}**")

        csv = df_results.to_csv(index=True).encode('utf-8')
        st.download_button("📅 Download Results", csv, "results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
