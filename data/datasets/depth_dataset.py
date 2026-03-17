import os
import random
import torch # Thêm import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

class DepthDataset(Dataset):
    def __init__(self, frames_dir, depth_dir, is_train=True):
        self.samples = []
        self.is_train = is_train

        for label in ["real_seq", "fake_seq"]:
            frames_label_dir = os.path.join(frames_dir, label)
            depth_label_dir = os.path.join(depth_dir, label)

            if not os.path.isdir(frames_label_dir) or not os.path.isdir(depth_label_dir):
                continue

            # --- MỚI: Đánh dấu label để tạo cờ is_fake ---
            is_fake = 1.0 if label == "fake_seq" else 0.0

            for seq_name in os.listdir(frames_label_dir):
                frames_seq_dir = os.path.join(frames_label_dir, seq_name)
                depth_seq_dir = os.path.join(depth_label_dir, seq_name)

                if not os.path.isdir(frames_seq_dir) or not os.path.isdir(depth_seq_dir):
                    continue

                for frame_name in os.listdir(frames_seq_dir):
                    frame_path = os.path.join(frames_seq_dir, frame_name)
                    depth_path = os.path.join(depth_seq_dir, frame_name)

                    if os.path.exists(frame_path) and os.path.exists(depth_path):
                        # --- MỚI: Lưu thêm is_fake vào tuple ---
                        self.samples.append((frame_path, depth_path, is_fake))

        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )
        self.blur = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # --- MỚI: Unpack thêm is_fake ---
        rgb_path, depth_path, is_fake = self.samples[idx]

        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")

        rgb = TF.resize(rgb, (128, 128))
        depth = TF.resize(depth, (128, 128))

        if self.is_train:
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)
                depth = TF.hflip(depth)
                
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                rgb = TF.rotate(rgb, angle)
                depth = TF.rotate(depth, angle)

            if random.random() > 0.2:
                rgb = self.color_jitter(rgb)
                
            if random.random() > 0.7:
                rgb = self.blur(rgb)

        rgb = TF.to_tensor(rgb)
        depth = TF.to_tensor(depth)

        if depth.max() > 1:
            depth = depth / 255.0

        # --- MỚI: Trả về thêm is_fake dạng Tensor ---
        return rgb, depth, torch.tensor(is_fake, dtype=torch.float32)