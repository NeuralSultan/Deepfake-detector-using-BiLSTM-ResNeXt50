import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import tempfile
from torchvision import transforms, models
from ultralytics import YOLO
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 60
BUFFER_SIZE = 5
IMG_SIZE = 224
FEATURE_DIM = 2048

LSTM_MODEL_PATH = #r"put your path here"
YOLO_MODEL_PATH = #r"put your path here"

st.set_page_config(page_title="Deepfake Detection System", layout="wide")

st.markdown("""
<div style='text-align: center; padding: 15px;'>
    <h1 style='color:#00BFFF;'> Deepfake Detection System</h1>
    <p style='font-size:18px;'>AI-powered Video Authenticity Analysis (YOLO + ResNeXt + BiLSTM)</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Settings")

    THRESHOLD = st.slider("Fake Detection Threshold", 0.0, 1.0, 0.72)

    st.markdown("---")
    st.markdown("###  Pipeline")
    st.write("""
    - YOLO: Face Detection  
    - ResNeXt: Feature Extraction  
    - BiLSTM: Temporal Classification  
    """)

    st.markdown("---")
    st.info("Upload a video and let AI analyze authenticity")

# LOAD MODELS
@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_MODEL_PATH)

    resnext = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
    resnext.fc = nn.Identity()
    resnext = resnext.to(DEVICE).eval()

    class LSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(FEATURE_DIM, 96, num_layers=1,
                                batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(192, 2)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out)

    lstm = LSTMClassifier().to(DEVICE)
    lstm.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
    lstm.eval()

    return yolo, resnext, lstm


yolo_model, resnext, lstm_model = load_models()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# UPLOAD 
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov"])

with col2:
    st.info("Supported formats: MP4 / AVI / MOV\nRecommended: short clips (≤30s)")

if uploaded_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.markdown("## 📊 Processing Dashboard")

    progress_bar = st.progress(0)
    frame_counter = st.empty()

    frames_seq = []
    seq_probs = []
    frames = []
    boxes = []

    buffer = []
    last_box = None
    frame_count = 0

    # PROCESS VIDEO 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb.copy())

        results = yolo_model(frame_rgb)[0]

        if len(results.boxes) > 0:
            box = results.boxes[0].xyxy[0].cpu().numpy().astype(int)
            buffer.append(box)

            if len(buffer) > BUFFER_SIZE:
                buffer.pop(0)

            x1, y1, x2, y2 = np.mean(buffer, axis=0).astype(int)
            last_box = (x1, y1, x2, y2)

        elif last_box is not None:
            x1, y1, x2, y2 = last_box
        else:
            boxes.append(None)
            continue

        boxes.append((x1, y1, x2, y2))

        face = frame_rgb[y1:y2, x1:x2]

        if face.size == 0:
            boxes[-1] = None
            continue

        img = Image.fromarray(face)
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = resnext(tensor).squeeze(0)

        frames_seq.append(feat)

        if len(frames_seq) == SEQ_LEN:
            seq_input = torch.stack(frames_seq).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = lstm_model(seq_input)
                prob = torch.softmax(out, dim=1).cpu().numpy()[0]

            seq_probs.extend([prob] * SEQ_LEN)
            frames_seq = []

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
        frame_counter.text(f"Processing frame {frame_count}/{total_frames}")

    cap.release()

    # FINAL SEQUENCE 
    if len(frames_seq) > 0:
        seq_input = torch.stack(frames_seq).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = lstm_model(seq_input)
            prob = torch.softmax(out, dim=1).cpu().numpy()[0]

        seq_probs.extend([prob] * len(frames_seq))

    # RESULTS
    avg_probs = np.mean(seq_probs, axis=0)

    real_prob = float(avg_probs[0])
    fake_prob = float(avg_probs[1])

    final_label = "Fake" if fake_prob > THRESHOLD else "Real"

    # RESULT CARDS
    st.markdown("## Analysis Result")

    col1, col2 = st.columns(2)

    col1.metric("Real Probability", f"{real_prob:.2f}")
    col2.metric("Fake Probability", f"{fake_prob:.2f}")

    st.progress(fake_prob)

    if final_label == "Fake":
        st.markdown(f"""
        <div style='background:#ff4b4b;padding:20px;border-radius:10px;text-align:center;'>
            <h2 style='color:white;'>FAKE VIDEO DETECTED</h2>
            <p style='color:white;'>Confidence: {fake_prob:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#00c853;padding:20px;border-radius:10px;text-align:center;'>
            <h2 style='color:white;'>REAL VIDEO</h2>
            <p style='color:white;'>Confidence: {real_prob:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    #  VIDEO OUTPUT 
    st.markdown("## Video Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Video")
        st.video(video_path)

    # GRAPH 
    st.markdown("## Fake Probability Over Time")

    probs = [p[1] for p in seq_probs if isinstance(p, (list, np.ndarray))]
    if len(probs) > 0:
        st.line_chart(probs)

    # FOOTER 
    st.markdown("---")
    st.caption("Deepfake Detection System | Graduation Project | AI Pipeline: YOLO + ResNeXt + BiLSTM")
