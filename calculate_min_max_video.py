import os
import cv2
import torch
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from models.unet_depth_cnn import UNetDepthCNN
from models.rppg_rnn import RPPG_RNN

# ======== Config ========
REAL_FOLDER = r"D:\project_new\data\train\real"
FAKE_FOLDER = r"D:\project_new\data\train\fake"
SEQ_LEN = 100
OUTPUT_JSON = "min_max.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Load models ========
depth_model = UNetDepthCNN().to(device)
depth_model.load_state_dict(torch.load(r"D:\project_new\checkpoints\unet_depth_epoch20.pth", map_location=device))
depth_model.eval()

rppg_model = RPPG_RNN().to(device)
rppg_model.load_state_dict(torch.load(r"D:\project_new\checkpoints_rppg\rppg_epoch5.pth", map_location=device))
rppg_model.eval()

# ======== Transform ========
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ======== Extract frames ========
def extract_frames(video_path, num_frames=SEQ_LEN):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = transform(img)
        frames.append(tensor)
    cap.release()

    if len(frames) < num_frames:
        raise ValueError(f"Not enough frames in {video_path}")

    return torch.stack(frames).unsqueeze(0).to(device)

# ======== Predict raw scores ========
@torch.no_grad()
def predict_raw_scores(seq_tensor):
    rppg = rppg_model(seq_tensor)
    rppg_score = torch.norm(rppg, p=2) ** 2

    first_frame = seq_tensor[0, 0]
    depth = depth_model(first_frame.unsqueeze(0))
    depth_score = torch.norm(depth, p=2) ** 2

    return rppg_score.item(), depth_score.item()

# ======== Main ========
def collect_scores(folder):
    scores = []
    for file in tqdm(os.listdir(folder), desc=os.path.basename(folder)):
        path = os.path.join(folder, file)
        if not path.lower().endswith((".mp4", ".avi", ".mov")):
            continue
        try:
            seq = extract_frames(path)
            r_score, d_score = predict_raw_scores(seq)
            scores.append((r_score, d_score))
        except Exception as e:
            print(f"Lỗi {file}: {e}")
    return scores

if __name__ == "__main__":
    print("Đang tính min/max rPPG và Depth từ video...")

    real_scores = collect_scores(REAL_FOLDER)
    fake_scores = collect_scores(FAKE_FOLDER)

    all_rppg = [r for r, _ in real_scores + fake_scores]
    all_depth = [d for _, d in real_scores + fake_scores]

    rppg_min = float(np.min(all_rppg))
    rppg_max = float(np.max(all_rppg))
    depth_min = float(np.min(all_depth))
    depth_max = float(np.max(all_depth))

    result = {
        "rppg_min": round(rppg_min, 4),
        "rppg_max": round(rppg_max, 4),
        "depth_min": round(depth_min, 4),
        "depth_max": round(depth_max, 4)
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=4)

    print("\nMIN/MAX đã được ghi vào min_max.json:")
    for k, v in result.items():
        print(f"{k}: {v}")
