import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

class DepthDataset(Dataset):
    def __init__(self, frames_dir, depth_dir, is_train=True):
        """
        Args:
            frames_dir: Đường dẫn thư mục ảnh RGB
            depth_dir: Đường dẫn thư mục Depth map
            is_train: Bật True khi train (để dùng Augmentation), False khi test/val.
        """
        self.samples = []
        self.is_train = is_train

        for label in ["real_seq", "fake_seq"]:
            frames_label_dir = os.path.join(frames_dir, label)
            depth_label_dir = os.path.join(depth_dir, label)

            if not os.path.isdir(frames_label_dir) or not os.path.isdir(depth_label_dir):
                continue

            for seq_name in os.listdir(frames_label_dir):
                frames_seq_dir = os.path.join(frames_label_dir, seq_name)
                depth_seq_dir = os.path.join(depth_label_dir, seq_name)

                if not os.path.isdir(frames_seq_dir) or not os.path.isdir(depth_seq_dir):
                    continue

                for frame_name in os.listdir(frames_seq_dir):
                    frame_path = os.path.join(frames_seq_dir, frame_name)
                    depth_path = os.path.join(depth_seq_dir, frame_name)

                    if os.path.exists(frame_path) and os.path.exists(depth_path):
                        self.samples.append((frame_path, depth_path))

        # Khai báo các phép biến đổi màu sắc (Chỉ dùng cho RGB)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )
        self.blur = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]

        # Load ảnh dưới dạng PIL
        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")

        # 1. Resize (BẮT BUỘC cho cả hai)
        rgb = TF.resize(rgb, (128, 128))
        depth = TF.resize(depth, (128, 128))

        # 2. AUGMENTATION (Chỉ chạy khi is_train = True)
        if self.is_train:
            # --- Biến đổi Không Gian (Áp dụng cho CẢ HAI) ---
            
            # Lật ngang (Xác suất 50%)
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)
                depth = TF.hflip(depth)
                
            # Xoay ngẫu nhiên từ -15 đến 15 độ (Xác suất 50%)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                rgb = TF.rotate(rgb, angle)
                depth = TF.rotate(depth, angle)

            # --- Biến đổi Photometric (CHỈ áp dụng cho RGB) ---
            
            # Đổi màu/độ sáng ngẫu nhiên (Xác suất 80%)
            if random.random() > 0.2:
                rgb = self.color_jitter(rgb)
                
            # Làm mờ ảnh giả lập camera out nét (Xác suất 30%)
            if random.random() > 0.7:
                rgb = self.blur(rgb)

        # 3. Chuyển thành Tensor
        rgb = TF.to_tensor(rgb)
        depth = TF.to_tensor(depth)

        # Normalize depth map nếu là ảnh uint8
        if depth.max() > 1:
            depth = depth / 255.0

        return rgb, depth