import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import soundfile as sf
import os
import tempfile
from sklearn.preprocessing import StandardScaler
import librosa

# Define CNN model
class VoiceCNN(nn.Module):
    def __init__(self):
        super(VoiceCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 3, 64)  # <== MODIFICAT: 64 * 4 = 256 (cum era în modelul salvat)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model and scaler
model = VoiceCNN()
model.load_state_dict(torch.load('voice_cnn.pth', map_location='cpu'))
model.eval()
scaler = joblib.load('scaler.pkl')

st.title("Parkinson’s Detection from Voice")
st.write("Upload a .wav file of a sustained vowel sound (e.g., 'ah')")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

if uploaded_file is not None:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Feature extraction
    features = extract_features(tmp_path)
    features = scaler.transform([features])
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # add channel dim

    # Prediction
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)

    label = "Parkinson's Positive" if predicted.item() == 1 else "Parkinson's Negative"
    st.write(f"### Prediction: {label}")
