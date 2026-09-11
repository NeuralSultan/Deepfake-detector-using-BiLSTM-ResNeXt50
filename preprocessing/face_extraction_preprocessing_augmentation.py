import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import random

# ----------- الإعدادات -----------
DATASET_DIRS = {
    "Original": r"D:\Dataset\Faceforensics++\FaceForensics++_C23\original",
    "DeepFakes": r"D:\Dataset\Faceforensics++\FaceForensics++_C23\Deepfakes"
}

OUTPUT_DIR = r"F:\Grad Project\output_sequences"
MODEL_PATH = r"F:\Grad Project\dataset_faces\model.pt"

SEQ_LEN = 60  # ممكن تغيره 35-50
IMG_SIZE = 224
MAX_SAMPLING_RATE = 3

# ----------- تحميل YOLO -----------
yolo_model = YOLO(MODEL_PATH)

# ----------- Resize بدون distortion -----------
def resize_no_padding(img, size=IMG_SIZE):
    return cv2.resize(img, (size, size))

# ----------- augmentation sequence-wise -----------
def augment_sequence_params():
    return {
        "flip": random.random() < 0.5,
        "alpha": 1 + random.uniform(-0.1, 0.1),
        "beta": random.randint(-10, 10),
        "blur": random.random() < 0.1,
        "noise": random.random() < 0.1
    }

def apply_augment(img, params):
    if params["flip"]:
        img = cv2.flip(img, 1)
    img = cv2.convertScaleAbs(img, alpha=params["alpha"], beta=params["beta"])
    if params["blur"]:
        img = cv2.GaussianBlur(img, (3,3), 0)
    if params["noise"]:
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    return img

# ----------- adaptive sampling rate based on video length -----------
def get_sampling_rate(total_frames):
    if total_frames < SEQ_LEN * MAX_SAMPLING_RATE:
        rate = max(1, total_frames // SEQ_LEN)
    else:
        rate = MAX_SAMPLING_RATE
    return rate

# ----------- معالجة فيديو -----------
def process_video(video_path, out_dir, is_real=True):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampling_rate = get_sampling_rate(total_frames)

    frame_count = 0
    seq_count = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sampling_rate != 0:
            frame_count += 1
            continue

        # --------- YOLO للوجه ---------
        results = yolo_model(frame)[0]
        if len(results.boxes) != 1:
            frame_count += 1
            continue

        x1, y1, x2, y2 = results.boxes[0].xyxy[0].cpu().numpy().astype(int)
        face = frame[y1:y2, x1:x2]

        # --------- Resize ---------
        face = resize_no_padding(face)
        frames.append(face)
        frame_count += 1

        if len(frames) == SEQ_LEN:
            # --------- Save original sequence ---------
            seq_count += 1
            seq_folder = os.path.join(out_dir, f"seq_{seq_count:04d}")
            os.makedirs(seq_folder, exist_ok=True)
            for i, f in enumerate(frames):
                cv2.imwrite(os.path.join(seq_folder, f"frame_{i+1:04d}.jpg"), f)

            # --------- Augmented sequence ----------
            do_augment = False
            if is_real:
                do_augment = random.random() < 0.5  # 50% من Real فقط
            else:
                do_augment = random.random() < 0.15  # 15% فقط للـ Fake

            if do_augment:
                aug_params = augment_sequence_params()
                seq_count += 1
                seq_folder_aug = os.path.join(out_dir, f"seq_{seq_count:04d}")
                os.makedirs(seq_folder_aug, exist_ok=True)
                for i, f in enumerate(frames):
                    f_aug = apply_augment(f, aug_params)
                    cv2.imwrite(os.path.join(seq_folder_aug, f"frame_{i+1:04d}.jpg"), f_aug)

            frames = []

    cap.release()
    return seq_count

# ----------- معالجة كل الفولدرات -----------
def process_all():
    for label, folder in DATASET_DIRS.items():
        print(f"\n🔥 Processing {label}")
        for video_file in os.listdir(folder):
            if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
                continue
            video_path = os.path.join(folder, video_file)
            video_name = Path(video_file).stem
            out_video_folder = os.path.join(OUTPUT_DIR, label, video_name)
            os.makedirs(out_video_folder, exist_ok=True)
            print(f"Processing {video_name}...")
            seq_count = process_video(video_path, out_video_folder, is_real=(label=="Original"))
            print(f"Saved {seq_count} sequences for {video_name}")

# ----------- تشغيل -----------
process_all()