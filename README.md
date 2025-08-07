Cấu Trúc Thư Mục Dự Án: Face Anti-Spoofing

├── data
│   ├── __pycache__/
│   ├── depth_dataset.py        # Script định nghĩa Dataset cho mô hình Depth
│   ├── rppg_dataset.py         # Script định nghĩa Dataset cho mô hình RPPG
│   ├── depth_maps/             # Thư mục chứa các bản đồ độ sâu (depth map) cho tập huấn luyện
│   ├── frames/                 # Thư mục chứa các khung hình đã cắt từ video của tập huấn luyện
│   ├── frames_test/            # Thư mục chứa các khung hình đã cắt từ video của tập kiểm thử
│   ├── test/                   # Thư mục chứa video gốc của tập kiểm thử
│   └── train/                  # Thư mục chứa video gốc của tập huấn luyện
│
├── models          # Script định nghĩa kiến trúc mô hình Depth CNN
│   ├── rppg_rnn.py             # Script định nghĩa kiến trúc mô hình RPPG-RNN (CNN-LSTM)
│   └── unet_depth_cnn.py       # Script định nghĩa kiến trúc mô hình UNet cho việc ước tính độ sâu
│
├── calculate_min_max_video.py  # Script tính toán giá trị min/max score cho việc chuẩn hóa trên video
├── calculate_min_max_webcam.py # Script tính toán giá trị min/max score cho việc chuẩn hóa trên webcam
├── demo_webcam.py              # Script chính để chạy demo phát hiện giả mạo real-time qua webcam         # 
├── evaluate.py                 # Script để đánh giá hiệu suất của mô hình trên tập test
├── extract_frames.py           # Script để tách video thành các khung hình (frames)
├── gen_midas.py                # Script tạo bản đồ độ sâu (depth map) từ ảnh RGB sử dụng mô hình MiDaS
├── min_max_webcam.json         # Tệp JSON lưu giá trị min/max score cho webcam
├── min_max.json                # Tệp JSON lưu giá trị min/max score cho video
├── predict_video.py            # Script để dự đoán real/fake trên một file video offline
├── README.md                   # Tệp hướng dẫn và mô tả dự án 
├── requirements.txt            # Tệp liệt kê các thư viện Python cần thiết để chạy dự án
├── train_depth.py              # Script để huấn luyện mô hình ước tính độ sâu (Depth)
├── train_rppg.py               # Script để huấn luyện mô hình RPPG-RNN

Giải thích chi tiết
1. Thư mục data/
Thư mục này chứa tất cả dữ liệu cần thiết cho việc huấn luyện và đánh giá mô hình.

depth_dataset.py & rppg_dataset.py: Các lớp Dataset của PyTorch, giúp tải và tiền xử lý dữ liệu một cách hiệu quả cho từng mô hình.

depth_maps/ & depth_maps_test/: Lưu trữ các ảnh ground-truth về độ sâu, được tạo ra từ gen_midas.py.

frames/ & frames_test/: Kết quả sau khi chạy extract_frames.py, chứa các khuôn mặt đã được cắt và sẵn sàng để đưa vào mô hình.

train/ & test/: Chứa các video gốc được phân chia thành hai tập huấn luyện và kiểm thử.

2. Thư mục models/
Chứa các file định nghĩa kiến trúc mạng nơ-ron.

unet_depth_cnn.py: Định nghĩa kiến trúc UNet để dự đoán bản đồ độ sâu từ ảnh RGB.

rppg_rnn.py: Định nghĩa kiến trúc kết hợp CNN và LSTM để học tín hiệu rPPG từ chuỗi khung hình.

3. Các tệp kịch bản (Scripts) ở thư mục gốc
Đây là các tệp thực thi chính của dự án.

Huấn luyện:

train_depth.py: Chạy để huấn luyện mô hình UNet.

train_rppg.py: Chạy để huấn luyện mô hình RPPG-RNN.

Tiền xử lý & Chuẩn bị:

extract_frames.py: Bước đầu tiên, dùng để xử lý video gốc.

gen_midas.py: Tạo dữ liệu độ sâu cho việc huấn luyện mô hình UNet.

calculate_min_max_*.py: Các script này rất quan trọng, dùng để tính toán các giá trị chuẩn hóa (normalization) cho điểm số (score) của mô hình. Việc này đảm bảo điểm số từ hai nhánh Depth và RPPG có cùng một thang đo trước khi kết hợp.

Đánh giá & Demo:

evaluate.py: Đo lường các chỉ số (Accuracy, F1-score, AUC) trên tập test.

predict_video.py: Áp dụng mô hình đã huấn luyện lên một file video .mp4.

demo_webcam.py: Script quan trọng nhất để trình diễn sản phẩm, chạy mô hình với dữ liệu trực tiếp từ webcam.

4. Các tệp khác
*.pth: Các tệp checkpoint, lưu lại trọng số của mô hình đã được huấn luyện để có thể tái sử dụng mà không cần huấn luyện lại.

*.json: Lưu các giá trị cấu hình, trong trường hợp này là các giá trị min/max để chuẩn hóa điểm số.

requirements.txt: Rất quan trọng để người khác có thể cài đặt môi trường và chạy lại dự án của bạn một cách dễ dàng bằng lệnh pip install -r requirements.txt.