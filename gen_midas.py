import os
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

input_dir = Path("/content/drive/MyDrive/project/frames_test")
output_dir = Path("/content/drive/MyDrive/project/depth_maps_test")
output_dir.mkdir(parents=True, exist_ok=True)

# Labels bạn muốn xử lý
labels = ["real_seq", "fake_seq"]

for label in labels:
    input_label_dir = input_dir / label
    output_label_dir = output_dir / label
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # Nếu thư mục input không tồn tại thì bỏ qua
    if not input_label_dir.exists():
        print(f"⚠️ Không tìm thấy thư mục {input_label_dir}")
        continue

    # Lặp qua các sequence
    seq_list = sorted(os.listdir(input_label_dir))
    for seq in tqdm(seq_list, desc=f"Label {label}"):
        seq_in = input_label_dir / seq
        seq_out = output_label_dir / seq
        seq_out.mkdir(parents=True, exist_ok=True)

        img_files = sorted([f for f in os.listdir(seq_in) if f.endswith((".jpg", ".png"))])

        for fname in img_files:
            img_path = seq_in / fname
            save_path = seq_out / fname
            
            # Đọc ảnh gốc bằng OpenCV
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            if label == "real_seq":
                # ==========================================
                # LUỒNG 1: DÙNG MIDAS CHO NHÃN REAL
                # ==========================================
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                input_tensor = transform(img_rgb)[0].to(device)

                with torch.no_grad():
                    prediction = midas(input_tensor.unsqueeze(0))
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                    depth = prediction.cpu().numpy()
                    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                    depth_uint8 = depth_norm.astype(np.uint8)

                    # Save JPG nén chất lượng thấp
                    cv2.imwrite(str(save_path), depth_uint8, [cv2.IMWRITE_JPEG_QUALITY, 80])

            elif label == "fake_seq":
                # ==========================================
                # LUỒNG 2: GÁN MA TRẬN 0 CHO NHÃN FAKE
                # ==========================================
                h, w = img.shape[:2]
                
                # Tạo ma trận toàn số 0 (màu đen hoàn toàn) với cùng kích thước ảnh gốc
                depth_uint8 = np.zeros((h, w), dtype=np.uint8)
                
                # Save JPG nén chất lượng thấp
                cv2.imwrite(str(save_path), depth_uint8, [cv2.IMWRITE_JPEG_QUALITY, 80])

print("✅ Hoàn tất sinh depth maps với dung lượng tối ưu.")