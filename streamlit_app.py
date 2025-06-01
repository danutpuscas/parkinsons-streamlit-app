import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model('cnn_model.keras')
scaler = joblib.load('scaler.pkl')

# Feature names
features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
            'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
            'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

# Web UI
st.title("🧠 Parkinson's Disease Detection")
st.write("Enter 22 voice features below:")

input_data = []
for feature in features:
    val = st.number_input(feature, format="%.5f")
    input_data.append(val)

if st.button("Predict"):
    X_input = np.array(input_data).reshape(1, -1)
    X_scaled = scaler.transform(X_input)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    pred = model.predict(X_scaled)[0][0]

    if pred >= 0.5:
        st.error("⚠️ Parkinson Detected")
    else:
        st.success("✅ No Parkinson Detected")
