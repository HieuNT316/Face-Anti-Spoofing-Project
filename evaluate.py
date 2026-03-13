import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# ===== CUSTOM IMPORT =====
# Import class AntiSpoofPredictor mà bạn vừa tạo
from models.predictor import AntiSpoofPredictor
from data.datasets.rppg_dataset import RPPGDataset  # Đảm bảo đường dẫn này đúng với project của bạn

# ===== CONFIG =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RPPG_MODEL_PATH = r"D:\ml_course\project2_face_anti\rppg_epoch5.pth"
DEPTH_MODEL_PATH = r"D:\ml_course\project2_face_anti\unet_depth_epoch20.pth"
DATA_DIR = r"D:\ml_course\project2_face_anti\data\frames_test"
SEQ_LEN = 5
BATCH_SIZE = 1  # BẮT BUỘC = 1 vì hàm predict_score trong predictor đang xử lý dạng [1, T, C, H, W]

# ===== TRANSFORMS =====
# CHÚ Ý: Đã chỉnh sửa để khớp với config của file predictor.py (128x128, không Normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ===== LOAD DATASET =====
dataset = RPPGDataset(root_dir=DATA_DIR, transform=transform, seq_len=SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"📊 Loaded {len(dataset)} sequences for evaluation.")

# ===== LOAD PREDICTOR =====
print("⚙️ Initializing Predictor...")
predictor = AntiSpoofPredictor(
    depth_model_path=DEPTH_MODEL_PATH,
    rppg_model_path=RPPG_MODEL_PATH,
    seq_len=SEQ_LEN
)

# ===== INFERENCE =====
true_labels = []
pred_scores = []

# Đảm bảo các mô hình đang ở trạng thái eval
predictor.depth_model.eval()
predictor.rppg_model.eval()

with torch.no_grad():
    # Nhận toàn bộ batch vào một biến duy nhất để kiểm tra
    for batch in tqdm(dataloader, desc="🔍 Evaluating"):
        
        # Tự động bóc tách (unpack) dựa trên số lượng giá trị dataset trả về
        if len(batch) == 3:
            x, _, label = batch
        elif len(batch) == 2:
            x, label = batch
        else:
            x, label = batch[0], batch[-1] # Fallback lấy đầu và cuối

        x = x.to(DEVICE)          # Kích thước: (1, T, C, H, W)

        try:
            # Gọi trực tiếp hàm dự đoán từ predictor
            result = predictor.predict_score(x)
            
            # Lấy điểm số tổng hợp (total_score)
            score = result['total_score']
            
            pred_scores.append(score)
            
            # Đảm bảo label lấy đúng dạng int (xử lý an toàn cho cả Tensor và int thường)
            true_label_val = int(label.item()) if torch.is_tensor(label) else int(label)
            true_labels.append(true_label_val)
            
        except Exception as e:
            print(f"\n❌ Lỗi khi dự đoán sequence: {e}")
            continue

pred_scores = np.array(pred_scores)
true_labels = np.array(true_labels)

# ===== TÌM NGƯỠNG TỐI ƯU =====
def find_best_threshold(scores, labels, metric='f1'):
    best_score = -1
    best_thresh = 0.0

    # Duyệt qua các ngưỡng khả dĩ từ min đến max của tập score
    for t in np.linspace(np.min(scores), np.max(scores), 100):
        # Trong AntiSpoofPredictor, score > threshold => Real (1), ngược lại => Fake/Spoof (0)
        preds = (scores >= t).astype(int)
        
        if metric == 'f1':
            score = f1_score(labels, preds, zero_division=0)
        elif metric == 'acc':
            score = accuracy_score(labels, preds)
        else:
            continue

        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score

# ===== ĐÁNH GIÁ HIỆU SUẤT =====
best_thresh, best_f1 = find_best_threshold(pred_scores, true_labels, metric='f1')

# Phân loại dựa trên ngưỡng tối ưu vừa tìm được
pred_labels = (pred_scores >= best_thresh).astype(int)

acc = accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels, zero_division=0)
rec = recall_score(true_labels, pred_labels, zero_division=0)
f1 = f1_score(true_labels, pred_labels, zero_division=0)
auc = roc_auc_score(true_labels, pred_scores)

cm = confusion_matrix(true_labels, pred_labels)

# ===== KẾT QUẢ =====
print("\n🎯 Evaluation Results:")
print(f"📌 Predictor Default Threshold : {predictor.threshold:.4f}")
print(f"📌 Optimal Threshold (Testset) : {best_thresh:.4f}")
print("-" * 30)
print(f"✅ Accuracy   : {acc:.4f}")
print(f"✅ Precision  : {prec:.4f}")
print(f"✅ Recall     : {rec:.4f}")
print(f"✅ F1-score   : {f1:.4f}")
print(f"✅ ROC-AUC    : {auc:.4f}")

# IN CONFUSION MATRIX
print("\n📊 Confusion Matrix:")
print(f"                     [Predicted Spoof (0)]  [Predicted Real (1)]")
if cm.shape == (2, 2):
    print(f"[Actual Spoof (0)] :          {cm[0][0]:<10} |          {cm[0][1]}")
    print(f"[Actual Real (1)]  :          {cm[1][0]:<10} |          {cm[1][1]}")
else:
    print(cm)

# VẼ VÀ LƯU ẢNH CONFUSION MATRIX
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Spoof (0)", "Real (1)"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Thresh = {best_thresh:.4f})")
plt.tight_layout()

# Tạo thư mục lưu kết quả nếu chưa có
output_dir = r"D:\project2_face_anti\eval_results"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "confusion_matrix.png")

plt.savefig(save_path, dpi=300)
print(f"\n📸 Đã lưu ảnh đồ thị tại: {save_path}")