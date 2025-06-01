import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib

# Model definition
class VoiceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.pool  = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.fc1   = nn.Linear(64 * 8, 64)
        self.fc2   = nn.Linear(64, 1)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(torch.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))

# Load model and scaler
model = VoiceCNN()
model.load_state_dict(torch.load('voice_cnn.pth', map_location='cpu'))
model.eval()
scaler = joblib.load('scaler.pkl')

# Feature names
features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
            'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
            'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

st.title("🧠 Parkinson's Detection from Voice (PyTorch)")
st.write("Enter 22 voice features below:")

input_data = [st.number_input(f, format="%.5f") for f in features]

if st.button("Predict"):
    X = np.array(input_data).reshape(1, -1)
    Xs = scaler.transform(X)
    Xt = torch.tensor(Xs, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        pred = model(Xt).item()
    if pred >= 0.5:
        st.error(f"⚠️ Parkinson Detected (score={pred:.2f})")
    else:
        st.success(f"✅ No Parkinson Detected (score={pred:.2f})")
