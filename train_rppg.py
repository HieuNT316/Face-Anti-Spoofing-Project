import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Thêm path để import được module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.datasets.rppg_dataset import RPPGDataset
from models.rppg_rnn import RPPG_RNN

# ---------------------------------------------------------
# 1. ĐỊNH NGHĨA CLASS EARLY STOPPING
# ---------------------------------------------------------
class EarlyStopping:
    """Ngừng training sớm nếu loss không cải thiện sau một số epoch nhất định."""
    def __init__(self, patience=5, min_delta=0.0001):
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

# Config
epochs_to_train = 10  # Số epoch bạn muốn train THÊM trong mỗi lần chạy
batch_size = 4
lr = 1e-4

# Thư mục lưu checkpoint
save_dir = "checkpoints\\rppg"
os.makedirs(save_dir, exist_ok=True)
latest_checkpoint_path = os.path.join(save_dir, "latest_rppg.pth")

# Dataset
dataset = RPPGDataset(root_dir=r"D:\ml_course\project2_face_anti\data\frames", seq_len=100)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = RPPG_RNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ---------------------------------------------------------
# 2. KHỞI TẠO EARLY STOPPING & RESUME TRAINING
# ---------------------------------------------------------
early_stopping = EarlyStopping(patience=7, min_delta=1e-4)
start_epoch = 0

if os.path.exists(latest_checkpoint_path):
    print(f"🔄 Tìm thấy checkpoint! Đang load từ: {latest_checkpoint_path}")
    checkpoint = torch.load(latest_checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
    # Cộng dồn số epoch muốn train thêm vào start_epoch
    num_epochs = start_epoch + epochs_to_train
    
    # Khôi phục trạng thái Early Stopping cũ (nếu có)
    if 'best_loss' in checkpoint:
        early_stopping.best_loss = checkpoint['best_loss']
        
    print(f"✅ Đã load thành công. Tiếp tục huấn luyện từ Epoch {start_epoch + 1}...")
else:
    print("✨ Không tìm thấy checkpoint cũ. Bắt đầu huấn luyện từ đầu...")
    num_epochs = epochs_to_train

# ---------------------------------------------------------
# VÒNG LẶP HUẤN LUYỆN
# ---------------------------------------------------------
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for video_seq, labels in loop:
        video_seq, labels = video_seq.to(device), labels.to(device)

        outputs = model(video_seq)                   # (B, 2)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=correct / total * 100)

    acc = correct / total * 100
    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    # ---------------------------------------------------------
    # 3. KIỂM TRA ĐIỀU KIỆN EARLY STOPPING & LƯU MODEL
    # ---------------------------------------------------------
    early_stopping(avg_loss)
    
    # Đóng gói trạng thái hiện tại
    save_state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'best_loss': early_stopping.best_loss
    }
    
    # Lưu file theo từng epoch và ghi đè file latest
    model_path = os.path.join(save_dir, f"rppg_epoch{epoch+1}.pth")
    torch.save(save_state, model_path)
    torch.save(save_state, latest_checkpoint_path)

    # Nếu trigger được kích hoạt, thoát khỏi vòng lặp
    if early_stopping.early_stop:
        print(f"🛑 Mức Loss không cải thiện sau {early_stopping.patience} epochs. Kích hoạt Early Stopping để dừng sớm!")
        break