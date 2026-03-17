import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class RPPGDataset(Dataset):
    def __init__(self, root_dir, seq_len=100, stride=50, transform=None):
        self.samples = []
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        # Hỗ trợ cả tên thư mục cũ và mới
        valid_labels = ['real', 'fake', 'real_seq', 'fake_seq']

        for label_dir in os.listdir(root_dir):
            if label_dir not in valid_labels:
                continue

            label_path = os.path.join(root_dir, label_dir)
            label_idx = 1 if 'real' in label_dir else 0

            # Duyệt qua từng thư mục video (vd: video_01, video_02...)
            for video_folder in os.listdir(label_path):
                video_path = os.path.join(label_path, video_folder)
                if not os.path.isdir(video_path):
                    continue

                # Lấy danh sách ảnh và sắp xếp chuẩn
                frame_files = sorted([
                    f for f in os.listdir(video_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])
                
                num_frames = len(frame_files)

                # ======================================================
                # THUẬT TOÁN SLIDING WINDOW ẢO (TRÊN RAM)
                # ======================================================
                if num_frames >= seq_len:
                    # Tạo các vị trí bắt đầu (start_idx) với bước nhảy stride
                    for start_idx in range(0, num_frames - seq_len + 1, self.stride):
                        # Thay vì lưu ảnh, ta chỉ lưu "bản đồ" vị trí
                        self.samples.append((video_path, frame_files, start_idx, label_idx))
                else:
                    print(f"⚠️ Bỏ qua {video_folder}: Chỉ có {num_frames} frames (cần tối thiểu {seq_len})")

        print(f"📊 [INFO] Đã tạo ảo {len(self.samples)} sequences từ dữ liệu gốc.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Unpack "bản đồ"
        video_path, frame_files, start_idx, label = self.samples[idx]
        
        frames = []
        # Chỉ load đúng seq_len frame tính từ vị trí start_idx
        for j in range(self.seq_len):
            frame_name = frame_files[start_idx + j]
            path = os.path.join(video_path, frame_name)
            
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            frames.append(img)
            
        video_tensor = torch.stack(frames)  # [seq_len, 3, H, W]
        return video_tensor, torch.tensor(label, dtype=torch.long)