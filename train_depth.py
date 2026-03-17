import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# --- MỚI: Import thư viện SSIM ---
# Cài đặt: pip install pytorch-msssim
from pytorch_msssim import ssim

from data.datasets.depth_dataset import DepthDataset
from models.unet_depth_cnn import UNetDepthCNN

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0  
        else:
            self.counter += 1
            print(f"  -> EarlyStopping: Chờ {self.counter}/{self.patience} epoch")
            if self.counter >= self.patience:
                self.early_stop = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DepthDataset(
    frames_dir=r"D:\ml_course\project2_face_anti\data\frames",
    depth_dir=r"D:\ml_course\project2_face_anti\data\depth_maps",
    is_train=True
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

model = UNetDepthCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

checkpoint_dir = "/content/drive/MyDrive/project2_face_anti/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_unet.pth")

start_epoch = 0
num_epochs = 10
early_stopping = EarlyStopping(patience=7, min_delta=1e-4) 

if os.path.exists(latest_checkpoint_path):
    print(f"🔄 Tìm thấy checkpoint! Đang load từ: {latest_checkpoint_path}")
    checkpoint = torch.load(latest_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    num_epochs += start_epoch 
    if 'best_loss' in checkpoint:
        early_stopping.best_loss = checkpoint['best_loss']
    print(f"✅ Đã load thành công. Tiếp tục huấn luyện từ Epoch {start_epoch + 1}...")
else:
    print("✨ Không tìm thấy checkpoint cũ. Bắt đầu huấn luyện từ đầu...")

# ========================================================
# CẤU HÌNH TRỌNG SỐ LOSS
# ========================================================
WEIGHT_FAKE = 5.0  # Phạt lỗi trên ảnh fake nặng gấp 5 lần
ALPHA = 0.5        # Tỷ lệ cân bằng giữa MSE và SSIM (0.5 cho mỗi bên)

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    # --- MỚI: Unpack thêm is_fakes ---
    for images, targets, is_fakes in pbar:
        images = images.to(device)
        targets = targets.to(device)
        is_fakes = is_fakes.to(device)

        outputs = model(images)
        
        # ---------------------------------------------------------
        # TÍNH TOÁN WEIGHTED LOSS & SSIM
        # ---------------------------------------------------------
        
        # 1. TÍNH MSE (Chưa lấy trung bình vội để nhân trọng số)
        mse_loss = nn.functional.mse_loss(outputs, targets, reduction='none')
        mse_loss_per_img = mse_loss.mean(dim=(1, 2, 3)) # Lỗi cho từng ảnh trong batch
        
        # Tạo vector trọng số: Nếu is_fakes == 1 thì lấy 5.0, ngược lại lấy 1.0
        weights = torch.where(
            is_fakes == 1.0, 
            torch.tensor(WEIGHT_FAKE, device=device), 
            torch.tensor(1.0, device=device)
        )
        
        # Tính MSE đã nhân trọng số
        weighted_mse = (mse_loss_per_img * weights).mean()
        
        # 2. TÍNH SSIM LOSS
        # data_range=1.0 vì ảnh đã chuẩn hóa về [0, 1]
        ssim_val = ssim(outputs, targets, data_range=1.0, size_average=True)
        loss_ssim = 1.0 - ssim_val 
        
        # 3. TỔNG HỢP LOSS
        loss = (ALPHA * loss_ssim) + ((1.0 - ALPHA) * weighted_mse)
        # ---------------------------------------------------------

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", ssim=f"{ssim_val.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch+1}] Avg Loss (Weighted + SSIM): {avg_loss:.4f}")

    early_stopping(avg_loss)
    
    save_state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'best_loss': early_stopping.best_loss
    }
    
    epoch_path = os.path.join(checkpoint_dir, f"unet_depth_epoch{epoch+1}.pth")
    torch.save(save_state, epoch_path)
    torch.save(save_state, latest_checkpoint_path)

    if early_stopping.early_stop:
        print(f"🛑 Mức Loss không cải thiện sau {early_stopping.patience} epochs. Kích hoạt Early Stopping để dừng sớm!")
        break