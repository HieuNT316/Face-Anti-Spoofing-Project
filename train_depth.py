import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from data.datasets.depth_dataset import DepthDataset
from models.unet_depth_cnn import UNetDepthCNN

# ---------------------------------------------------------
# 1. ĐỊNH NGHĨA CLASS EARLY STOPPING
# ---------------------------------------------------------
class EarlyStopping:
    """Ngừng training sớm nếu loss không cải thiện sau một số epoch nhất định."""
    def __init__(self, patience=5, min_delta=0.0001):
        """
        Args:
            patience (int): Số epoch chờ đợi nếu loss không giảm.
            min_delta (float): Mức giảm tối thiểu để được coi là có cải thiện.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0  # Reset bộ đếm nếu loss giảm tốt
        else:
            self.counter += 1
            print(f"  -> EarlyStopping: Chờ {self.counter}/{self.patience} epoch")
            if self.counter >= self.patience:
                self.early_stop = True

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = DepthDataset(
    frames_dir=r"D:\ml_course\project2_face_anti\data\frames",
    depth_dir=r"D:\ml_course\project2_face_anti\data\depth_maps"
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Khởi tạo model và optimizer
model = UNetDepthCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Thư mục checkpoint trên Drive
checkpoint_dir = "/content/drive/MyDrive/project2_face_anti/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_unet.pth")

start_epoch = 0
num_epochs = 10

# ---------------------------------------------------------
# 2. KHỞI TẠO EARLY STOPPING
# ---------------------------------------------------------
# Dừng nếu loss không giảm quá 0.0001 trong 7 epoch liên tiếp
early_stopping = EarlyStopping(patience=7, min_delta=1e-4) 

# Load checkpoint nếu có
if os.path.exists(latest_checkpoint_path):
    print(f"🔄 Tìm thấy checkpoint! Đang load từ: {latest_checkpoint_path}")
    checkpoint = torch.load(latest_checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    num_epochs += start_epoch 
    
    # Khôi phục trạng thái Early Stopping cũ (nếu có)
    if 'best_loss' in checkpoint:
        early_stopping.best_loss = checkpoint['best_loss']
        
    print(f"✅ Đã load thành công. Tiếp tục huấn luyện từ Epoch {start_epoch + 1}...")
else:
    print("✨ Không tìm thấy checkpoint cũ. Bắt đầu huấn luyện từ đầu...")

# ---------------------------------------------------------
# VÒNG LẶP HUẤN LUYỆN
# ---------------------------------------------------------
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

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

    # ---------------------------------------------------------
    # 3. KIỂM TRA ĐIỀU KIỆN EARLY STOPPING
    # ---------------------------------------------------------
    early_stopping(avg_loss)
    
    # Lưu model
    save_state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'best_loss': early_stopping.best_loss # Lưu lại kỷ lục loss tốt nhất
    }
    
    epoch_path = os.path.join(checkpoint_dir, f"unet_depth_epoch{epoch+1}.pth")
    torch.save(save_state, epoch_path)
    torch.save(save_state, latest_checkpoint_path)

    # Nếu trigger được kích hoạt, thoát khỏi vòng lặp
    if early_stopping.early_stop:
        print(f"🛑 Mức Loss không cải thiện sau {early_stopping.patience} epochs. Kích hoạt Early Stopping để dừng sớm!")
        break