import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class RPPGDataset(Dataset):
    def __init__(self, root_dir, seq_len=100, transform=None):
        self.samples = []
        self.seq_len = seq_len
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        for label in ['real_seq', 'fake_seq']:
            label_path = os.path.join(root_dir, label)
            if not os.path.exists(label_path):
                continue

            for seq in os.listdir(label_path):
                seq_path = os.path.join(label_path, seq)
                if not os.path.isdir(seq_path):
                    continue

                # Lấy danh sách ảnh hợp lệ (jpg, png) và sắp xếp theo tên
                frame_files = sorted([
                    f for f in os.listdir(seq_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])

                # Duyệt theo cửa sổ trượt để lấy sequence
                for i in range(0, len(frame_files) - seq_len + 1):
                    frame_paths = [os.path.join(seq_path, frame_files[j]) for j in range(i, i + seq_len)]
                    self.samples.append((frame_paths, 1 if label == 'real_seq' else 0))

        print(f"[INFO] Tổng số sequence samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = []
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            frames.append(img)
        video_tensor = torch.stack(frames)  # [seq_len, 3, H, W]
        return video_tensor, torch.tensor(label, dtype=torch.long)
