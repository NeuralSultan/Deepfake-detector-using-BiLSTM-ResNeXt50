import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# SEED 
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# DATA PATH 
DATA_PATH = #r"put your path here"

# LOAD DATA
video_paths, video_labels = [], []

for class_name, label in [("Real", 0), ("Fake", 1)]:
    class_path = os.path.join(DATA_PATH, class_name)

    for video in os.listdir(class_path):
        v = os.path.join(class_path, video)
        if os.path.isdir(v) and len(os.listdir(v)) > 0:
            video_paths.append(v)
            video_labels.append(label)

video_paths = np.array(video_paths)
video_labels = np.array(video_labels)

# SPLIT 
train_videos, temp_videos, train_labels, temp_labels = train_test_split(
    video_paths, video_labels, test_size=0.3, stratify=video_labels, random_state=42
)

val_videos, test_videos, val_labels, test_labels = train_test_split(
    temp_videos, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

# DATASET 
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        v_path = self.video_paths[idx]
        label = self.labels[idx]

        seq_files = sorted([f for f in os.listdir(v_path) if f.endswith(".npy")])

        seqs = []
        for f in seq_files[:5]:
            seqs.append(np.load(os.path.join(v_path, f)))

        seq = np.mean(seqs, axis=0)

        # normalization
        seq = (seq - seq.mean()) / (seq.std() + 1e-6)

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label)


def collate_fn(batch):
    x, y = zip(*batch)
    return torch.stack(x), torch.tensor(y)

# DATALOADERS 
train_loader = DataLoader(VideoDataset(train_videos, train_labels),
                          batch_size=8, shuffle=True, collate_fn=collate_fn)

val_loader = DataLoader(VideoDataset(val_videos, val_labels),
                        batch_size=8, shuffle=False, collate_fn=collate_fn)

test_loader = DataLoader(VideoDataset(test_videos, test_labels),
                         batch_size=8, shuffle=False, collate_fn=collate_fn)

# MODEL 
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(96 * 2, 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)

        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        h = self.dropout(h)

        return self.fc(h)

# DEVICE 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier().to(device)

# LOSS + OPTIMIZE
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

# TRACKING 
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# TRAIN 
epochs = 15
best_val = 0
patience = 3
counter = 0

for epoch in range(epochs):

    # TRAIN 
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)

        loss = criterion(out, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # VALIDATION
    model.eval()
    preds_all, labels_all, probs_all = [], [], []
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()

            probs = torch.softmax(out, dim=1)[:, 1]
            preds = out.argmax(1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = accuracy_score(labels_all, preds_all)

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    
    if val_acc > best_val:
        best_val = val_acc
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(" Saved Best Model")
    else:
        counter += 1
        if counter >= patience:
            print(" Early Stopping Triggered")
            break


model.load_state_dict(torch.load("best_model.pth"))
model.eval()

preds_all, labels_all, probs_all = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        probs = torch.softmax(out, dim=1)[:, 1]
        preds = out.argmax(1)

        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(y.cpu().numpy())
        probs_all.extend(probs.cpu().numpy())

print("\n TEST RESULTS")
print("Acc:", accuracy_score(labels_all, preds_all))
print("Prec:", precision_score(labels_all, preds_all))
print("Recall:", recall_score(labels_all, preds_all))
print("F1:", f1_score(labels_all, preds_all))

