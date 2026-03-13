import os
import numpy as np
import cv2
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# --- Bandpass filter: 0.75–2.5 Hz tương đương nhịp tim 45–150 bpm ---
def bandpass_filter(signal, low=0.75, high=2.5, fs=30, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

# --- Chuẩn hóa về mean=0, std=1 ---
def normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

# --- Xử lý 1 sequence ---
def process_sequence(seq_path):
    frames = []
    for fname in sorted(os.listdir(seq_path)):
        fpath = os.path.join(seq_path, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        roi = cv2.resize(img, (36, 36))  # chuẩn hóa size
        green = roi[:, :, 1]
        frames.append(green)

    if len(frames) < 21:
        return None  # bỏ qua nếu quá ngắn

    frames = np.stack(frames)
    raw = np.mean(frames, axis=(1, 2))  # tín hiệu raw từ kênh xanh
    filtered = bandpass_filter(raw)
    normed = normalize(filtered)

    fft_feat = np.abs(np.fft.rfft(normed))
    return fft_feat[:50]  # lấy 50 thành phần đầu (low-freq)

# --- Main ---
input_root = "D:/ml_course/project2_face_anti/frames_test"
output_root = "D:/ml_course/project2_face_anti/rppg_signals"

os.makedirs(output_root, exist_ok=True)

for label in ['real_seq', 'fake_seq']:
    input_label_dir = os.path.join(input_root, label)
    output_label_dir = os.path.join(output_root, label)
    os.makedirs(output_label_dir, exist_ok=True)

    seq_names = sorted(os.listdir(input_label_dir))
    print(f"Processing {label}...")
    for seq_name in tqdm(seq_names):
        seq_path = os.path.join(input_label_dir, seq_name)
        fft_feat = process_sequence(seq_path)
        if fft_feat is None:
            print(f"⚠️ Skipped: {seq_name} (too short or unreadable)")
            continue
        save_path = os.path.join(output_label_dir, f"{seq_name}.npy")
        np.save(save_path, fft_feat)
        # print(f"✅ Saved: {save_path}")
