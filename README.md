# Cấu trúc Thư Mục Dự Án: Face Anti-Spoofing


```plaintext
Face Anti-Spoofing
├── data
│   ├── __pycache__/
│   ├── depth_dataset.py          # Dataset cho Depth model
│   ├── rppg_dataset.py           # Dataset cho RPPG model
│   ├── depth_maps/               # Depth maps cho tập huấn luyện
│   ├── frames/                   # Frames từ video (train)
│   ├── frames_test/              # Frames từ video (test)
│   ├── test/                     # Video gốc (test)
│   └── train/                    # Video gốc (train)
├── models
│   ├── rppg_rnn.py               # Kiến trúc RPPG-RNN (CNN-LSTM)
│   └── unet_depth_cnn.py         # Kiến trúc UNet Depth CNN
├── calculate_min_max_video.py    # Tính min/max score cho video
├── calculate_min_max_webcam.py   # Tính min/max score cho webcam
├── demo_webcam.py                # Demo real-time qua webcam
├── evaluate.py                   # Đánh giá mô hình
├── extract_frames.py             # Tách video thành frames
├── gen_midas.py                  # Tạo depth map từ ảnh RGB
├── min_max_webcam.json           # Giá trị min/max cho webcam
├── min_max.json                  # Giá trị min/max cho video
├── predict_video.py              # Dự đoán real/fake từ video
├── README.md                     # Tài liệu mô tả dự án
├── requirements.txt              # Danh sách thư viện
├── train_depth.py                # Huấn luyện Depth model
└── train_rppg.py                 # Huấn luyện RPPG model


---

## Giải thích chi tiết

**Thư mục `data/`**  
- `depth_dataset.py` & `rppg_dataset.py`: Các lớp Dataset của PyTorch để tải và tiền xử lý dữ liệu.  
- `depth_maps/` & `depth_maps_test/`: Ảnh ground-truth về độ sâu.  
- `frames/` & `frames_test/`: Ảnh khuôn mặt đã cắt từ video.  
- `train/` & `test/`: Video gốc chia theo tập huấn luyện và kiểm thử.  

**Thư mục `models/`**  
- `unet_depth_cnn.py`: UNet dự đoán bản đồ độ sâu.  
- `rppg_rnn.py`: CNN + LSTM học tín hiệu rPPG.  

**Các script chính**  
- **Huấn luyện:** `train_depth.py`, `train_rppg.py`  
- **Tiền xử lý:** `extract_frames.py`, `gen_midas.py`, `calculate_min_max_*.py`  
- **Đánh giá & Demo:** `evaluate.py`, `predict_video.py`, `demo_webcam.py`  

**Khác**  
- `*.pth`: Checkpoint trọng số mô hình.  
- `*.json`: Giá trị cấu hình chuẩn hóa.  
- `requirements.txt`: Môi trường Python.  
