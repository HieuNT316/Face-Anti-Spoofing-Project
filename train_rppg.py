import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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
    """Ngừng training sớm nếu Validation Loss không cải thiện sau một số epoch nhất định."""
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset bộ đếm nếu loss giảm tốt
        else:
            self.counter += 1
            print(f"  -> EarlyStopping: Chờ {self.counter}/{self.patience} epoch")
            if self.counter >= self.patience:
                self.early_stop = True

# ---------------------------------------------------------
# 2. THIẾT LẬP CƠ BẢN & HYPERPARAMETERS
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs_to_train = 10  # Số epoch muốn train THÊM trong mỗi lần chạy
batch_size = 4        # NẾU BỊ LỖI TRÀN RAM (OOM), HÃY GIẢM XUỐNG 2
lr = 1e-4

# Thư mục lưu checkpoint
save_dir = "checkpoints\\rppg"
os.makedirs(save_dir, exist_ok=True)
latest_checkpoint_path = os.path.join(save_dir, "latest_rppg.pth")
best_checkpoint_path = os.path.join(save_dir, "best_rppg.pth")

# ---------------------------------------------------------
# 3. CHUẨN BỊ DỮ LIỆU (TRAIN / VAL SPLIT)
# ---------------------------------------------------------
# Khởi tạo Dataset tổng
data_root = r"D:\ml_course\project2_face_anti\data\frames"
full_dataset = RPPGDataset(root_dir=data_root, seq_len=100, stride=50)

# Chia tỉ lệ 80% Train, 20% Val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Đảm bảo seed cố định nếu bạn muốn các lần chạy chia data giống hệt nhau
# generator = torch.Generator().manual_seed(42) 
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"📊 Dữ liệu: Tổng {len(full_dataset)} seqs | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ---------------------------------------------------------
# 4. KHỞI TẠO MODEL, LOSS & OPTIMIZER
# ---------------------------------------------------------
model = RPPG_RNN().to(device)

# THÊM TRỌNG SỐ CHO LOSS: Phạt nặng class 0 (Fake) gấp 3 lần class 1 (Real)
weights = torch.tensor([3.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=lr)

# ---------------------------------------------------------
# 5. KHÔI PHỤC CHECKPOINT (RESUME TRAINING)
# ---------------------------------------------------------
early_stopping = EarlyStopping(patience=7, min_delta=1e-4)
start_epoch = 0

if os.path.exists(latest_checkpoint_path):
    print(f"🔄 Tìm thấy checkpoint! Đang load từ: {latest_checkpoint_path}")
    checkpoint = torch.load(latest_checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    num_epochs = start_epoch + epochs_to_train
    
    if 'best_loss' in checkpoint:
        early_stopping.best_loss = checkpoint['best_loss']
        
    print(f"✅ Đã load thành công. Tiếp tục huấn luyện từ Epoch {start_epoch + 1}...")
else:
    print("✨ Không tìm thấy checkpoint cũ. Bắt đầu huấn luyện từ đầu...")
    num_epochs = epochs_to_train

# ---------------------------------------------------------
# 6. VÒNG LẶP HUẤN LUYỆN CHÍNH
# ---------------------------------------------------------
for epoch in range(start_epoch, num_epochs):
    # ==========================
    # PHẦN A: TRAINING
    # ==========================
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    for video_seq, labels in train_loop:
        video_seq, labels = video_seq.to(device), labels.to(device)

        outputs = model(video_seq)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

        train_loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total * 100

    # ==========================
    # PHẦN B: VALIDATION
    # ==========================
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        for val_seq, val_labels in val_loop:
            val_seq, val_labels = val_seq.to(device), val_labels.to(device)
            
            val_outputs = model(val_seq)
            v_loss = criterion(val_outputs, val_labels)
            
            val_loss += v_loss.item()
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += (val_preds == val_labels).sum().item()
            val_total += val_labels.size(0)
            
    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total * 100
    
    print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.1f}%) | Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.1f}%)")

    # ==========================
    # PHẦN C: EARLY STOPPING & LƯU MODEL
    # ==========================
    early_stopping(avg_val_loss)
    
    save_state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'best_loss': early_stopping.best_loss
    }
    
    # Lưu file latest (ghi đè liên tục)
    torch.save(save_state, latest_checkpoint_path)
    
    # Nếu vòng lặp này có Val Loss tốt nhất, lưu thành best_model
    if avg_val_loss <= early_stopping.best_loss:
        torch.save(save_state, best_checkpoint_path)
        print("  🌟 Đã lưu Best Model mới!")

    if early_stopping.early_stop:
        print(f"🛑 Val Loss không giảm sau {early_stopping.patience} epochs. Early Stopping kích hoạt!")
        break

print("🎉 Hoàn tất quá trình huấn luyện!")