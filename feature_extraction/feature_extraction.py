import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ---------------- إعدادات ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 60
IMG_SIZE = 224

FRAMES_DIR = r"F:\Grad Project\frames_sequence3"
FEATURES_DIR = r"F:\Grad Project\features_sequence3"

os.makedirs(FEATURES_DIR, exist_ok=True)

# ---------------- ResNeXt50 ----------------
resnext = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
resnext.fc = torch.nn.Identity()
resnext = resnext.to(DEVICE)
resnext.eval()

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ---------------- Feature Extraction ----------------
for cls in ["Real", "Fake"]:
    cls_path = os.path.join(FRAMES_DIR, cls)
    out_cls_path = os.path.join(FEATURES_DIR, cls)
    os.makedirs(out_cls_path, exist_ok=True)

    for video in os.listdir(cls_path):
        video_path = os.path.join(cls_path, video)
        out_video_path = os.path.join(out_cls_path, video)
        os.makedirs(out_video_path, exist_ok=True)

        for seq in os.listdir(video_path):
            seq_path = os.path.join(video_path, seq)
            save_path = os.path.join(out_video_path, f"{seq}.npy")

            frames = sorted([
                f for f in os.listdir(seq_path)
                if f.endswith(('.jpg', '.png'))
            ])

            if len(frames) < SEQ_LEN:
                continue  # سيب السيكونس الصغيرة

            features = []

            for f in frames[:SEQ_LEN]:
                img = Image.open(os.path.join(seq_path, f)).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    feat = resnext(img_tensor).squeeze(0)  # [2048]

                features.append(feat.cpu().numpy())

            features = np.stack(features)  # [SEQ_LEN, 2048]
            np.save(save_path, features)

print("✅ Feature Extraction Done!")