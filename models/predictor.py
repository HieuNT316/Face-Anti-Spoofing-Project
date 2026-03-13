import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Nhập các mô hình từ project của bạn
try:
    from .unet_depth_cnn import UNetDepthCNN
    from .rppg_rnn import RPPG_RNN
except ImportError as e:
    print(f"⚠️ Cảnh báo: Không thể import model. Lỗi: {e}")

class AntiSpoofPredictor:
    """
    Module xử lý logic dự đoán Face Anti-Spoofing kết hợp U-Net (Depth) và CNN-LSTM/RNN (rPPG).
    """
    def __init__(self, depth_model_path, rppg_model_path, seq_len=100, alpha=0.8, threshold=0.5356):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cấu hình siêu tham số
        self.seq_len = seq_len
        self.alpha = alpha
        self.threshold = threshold

        # Cấu hình chuẩn hóa (Min-Max)
        self.rppg_min, self.rppg_max = 0.0034, 0.5022
        self.depth_min, self.depth_max = 3388.8628, 8643.7754

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        # Khởi tạo và tải trọng số mô hình
        self.depth_model = None
        self.rppg_model = None
        self._load_models(depth_model_path, rppg_model_path)

    def _load_models(self, depth_path, rppg_path):
        try:
            print("🔍 Đang tải mô hình...")
            
            # 1. Tải mô hình Depth
            self.depth_model = UNetDepthCNN().to(self.device)
            depth_checkpoint = torch.load(depth_path, map_location=self.device)
            # Kiểm tra xem đây là file lưu kiểu mới (Dictionary) hay kiểu cũ (Raw Weights)
            if isinstance(depth_checkpoint, dict) and 'model_state_dict' in depth_checkpoint:
                self.depth_model.load_state_dict(depth_checkpoint['model_state_dict'])
            else:
                self.depth_model.load_state_dict(depth_checkpoint)
            self.depth_model.eval()

            # 2. Tải mô hình rPPG
            self.rppg_model = RPPG_RNN().to(self.device)
            rppg_checkpoint = torch.load(rppg_path, map_location=self.device)
            # Tương tự cho rPPG
            if isinstance(rppg_checkpoint, dict) and 'model_state_dict' in rppg_checkpoint:
                self.rppg_model.load_state_dict(rppg_checkpoint['model_state_dict'])
            else:
                self.rppg_model.load_state_dict(rppg_checkpoint)
            self.rppg_model.eval()
            
            print("✅ Đã tải mô hình thành công.")
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình: {e}. Hệ thống sẽ không thể dự đoán.")

    def extract_frames(self, video_path, progress_callback=None):
        """
        Trích xuất frames từ video. 
        Tham số progress_callback (tùy chọn) dùng để cập nhật thanh tiến trình trên GUI.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video tại đường dẫn: {video_path}")

        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_num_frames = min(self.seq_len, total_frames_in_video)
        indices = np.linspace(0, total_frames_in_video - 1, actual_num_frames, dtype=int)
        
        frames, raw_imgs = [], []
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_imgs.append(Image.fromarray(frame_rgb))
            
            img_tensor = self.transform(Image.fromarray(frame_rgb))
            frames.append(img_tensor)
            
            # Gọi callback để cập nhật tiến trình (nếu có)
            if progress_callback:
                progress_callback(i + 1, actual_num_frames)
                
        cap.release()
        
        if not frames:
            raise RuntimeError("Không trích xuất được frame nào từ video.")
            
        seq_tensor = torch.stack(frames).unsqueeze(0).to(self.device)
        return seq_tensor, raw_imgs

    @torch.no_grad()
    def predict_score(self, seq_tensor):
        """
        Thực hiện inference để tính toán điểm số từ 2 luồng.
        """
        if self.depth_model is None or self.rppg_model is None:
            raise RuntimeError("Mô hình chưa được tải. Không thể dự đoán.")

        # Tính toán điểm rPPG
        rppg_feat = self.rppg_model(seq_tensor)
        rppg_score = torch.norm(rppg_feat, p=2) ** 2

        # Tính toán điểm Depth (dùng frame đầu tiên)
        first_frame = seq_tensor[0, 0]  # [3, H, W]
        depth_map = self.depth_model(first_frame.unsqueeze(0))
        depth_score = torch.norm(depth_map, p=2) ** 2

        # Chuẩn hóa (Min-Max)
        rppg_norm = (rppg_score.item() - self.rppg_min) / (self.rppg_max - self.rppg_min + 1e-8)
        depth_norm = (depth_score.item() - self.depth_min) / (self.depth_max - self.depth_min + 1e-8)
        
        # Tổng hợp điểm số
        total_score = rppg_norm + self.alpha * depth_norm
        
        # Kết luận
        is_real = total_score > self.threshold

        return {
            "is_real": is_real,
            "verdict": "✅ REAL" if is_real else "❌ FAKE",
            "total_score": total_score,
            "rppg_score": rppg_score.item(),
            "depth_score": depth_score.item(),
            "rppg_norm": rppg_norm,
            "depth_norm": depth_norm
        }

    def analyze_video(self, video_path, progress_callback=None):
        """
        Hàm tiện ích gom chung cả bước trích xuất frame và dự đoán.
        """
        seq_tensor, raw_imgs = self.extract_frames(video_path, progress_callback)
        result = self.predict_score(seq_tensor)
        return result, raw_imgs