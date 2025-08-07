import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

# =========================
# CONFIG
# =========================
VIDEO_PATH = r"D:\project_new\demo4.mp4"  # 🔹 Thay đường dẫn video tại đây
SEQ_LEN = 100
ALPHA = 0.8
THRESHOLD = 0.5356  # 🔹 Threshold tối ưu đã tìm được

RPPG_MIN, RPPG_MAX = 0.0034, 0.5022
DEPTH_MIN, DEPTH_MAX = 3388.8628, 8643.7754

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform cho frame
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# =========================
# Import models
# =========================
from models.unet_depth_cnn import UNetDepthCNN
from models.rppg_rnn import RPPG_RNN

print("Loading Depth model (UNet)...")
depth_model = UNetDepthCNN().to(device)
depth_model.load_state_dict(torch.load(r"D:\project_new\checkpoints\unet_depth_epoch20.pth", map_location=device))
depth_model.eval()

print("Loading RPPG model...")
rppg_model = RPPG_RNN().to(device)
rppg_model.load_state_dict(torch.load(r"D:\project_new\checkpoints_rppg\rppg_epoch5.pth", map_location=device))
rppg_model.eval()

print("Models loaded.\n")

# =========================
# Hàm trích xuất frame từ video
# =========================
def extract_frames(video_path, num_frames=SEQ_LEN):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Không đọc được frame {idx} từ video {video_path}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tensor = transform(img_pil)
        frames.append(img_tensor)
    cap.release()

    return torch.stack(frames).unsqueeze(0).to(device)  # [1, T, 3, H, W]

# =========================
# Hàm tính điểm dự đoán
# =========================
@torch.no_grad()
def predict_score(seq_tensor):
    # 1. RPPG
    rppg_feat = rppg_model(seq_tensor)  # [1, 2]
    rppg_score = torch.norm(rppg_feat, p=2) ** 2

    # 2. Depth
    first_frame = seq_tensor[0, 0]  # [3, H, W]
    depth_map = depth_model(first_frame.unsqueeze(0))  # [1, 1, H, W]
    depth_score = torch.norm(depth_map, p=2) ** 2

    # === Chuẩn hóa ===
    rppg_norm = (rppg_score.item() - RPPG_MIN) / (RPPG_MAX - RPPG_MIN + 1e-8)
    depth_norm = (depth_score.item() - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN + 1e-8)
    total_score = rppg_norm + ALPHA * depth_norm

    return {
        "total_score": total_score,
        "rppg_score": rppg_score.item(),
        "depth_score": depth_score.item(),
        "rppg_norm": rppg_norm,
        "depth_norm": depth_norm
    }

# =========================
# Main
# =========================
if __name__ == "__main__":
    print(f"🎥 Đang xử lý video: {VIDEO_PATH}")
    seq_tensor = extract_frames(VIDEO_PATH, SEQ_LEN)
    result = predict_score(seq_tensor)

    print("\n CHI TIẾT:")
    print(f"Raw RPPG Score   : {result['rppg_score']:.2f}")
    print(f"Raw Depth Score  : {result['depth_score']:.2f}")
    print(f"Norm RPPG Score  : {result['rppg_norm']:.4f}")
    print(f"Norm Depth Score : {result['depth_norm']:.4f}")
    print(f"Total Score      : {result['total_score']:.4f}")

    if result['total_score'] > THRESHOLD:
        print("\nKết quả: REAL")
    else:
        print("\nKết quả: FAKE")