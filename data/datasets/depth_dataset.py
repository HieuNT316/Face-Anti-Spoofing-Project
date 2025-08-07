# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# import os

# class DepthDataset(Dataset):
#     def __init__(self, root_dir):
#         self.sequences = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
#                           if os.path.isdir(os.path.join(root_dir, d))]
#         self.transform_rgb = transforms.Compose([
#             transforms.Resize((128, 128)),
#             transforms.ToTensor()
#         ])
#         self.transform_depth = transforms.Compose([
#             transforms.Resize((128, 128)),
#             transforms.ToTensor()
#         ])

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         seq_path = self.sequences[idx]
#         rgb_path = os.path.join(seq_path, "frame_001.png")  # ảnh RGB đầu tiên
#         depth_path = os.path.join(seq_path, "depth_001.png")    # ảnh depth map

#         rgb = self.transform_rgb(Image.open(rgb_path).convert("RGB"))
#         depth = self.transform_depth(Image.open(depth_path).convert("L"))  # độ sâu là ảnh xám

#         return rgb, depth
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class DepthDataset(Dataset):
    def __init__(self, frames_dir, depth_dir):
        self.samples = []

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

        self.transform_rgb = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        self.transform_depth = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]

        rgb = self.transform_rgb(Image.open(rgb_path).convert("RGB"))
        depth = self.transform_depth(Image.open(depth_path).convert("L"))

        # Normalize depth map nếu là ảnh uint8
        if depth.max() > 1:
            depth = depth / 255.0

        return rgb, depth
