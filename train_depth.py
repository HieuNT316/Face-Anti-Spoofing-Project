import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from data.datasets.depth_dataset import DepthDataset
from models.unet_depth_cnn import UNetDepthCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = DepthDataset(
    frames_dir=r"D:\project_new\data\frames\frames",
    depth_dir=r"D:\project_new\data\depth_maps"
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Khởi tạo model
model = UNetDepthCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Tạo thư mục lưu checkpoint
os.makedirs("checkpoints", exist_ok=True)

# Huấn luyện
for epoch in range(20):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

    # Lưu model mỗi epoch
    torch.save(model.state_dict(), f"checkpoints/unet_depth_epoch{epoch+1}.pth")
