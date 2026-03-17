import os
import cv2
import shutil

def extract_frames(video_path, output_folder, max_frames=300, step=1):
    """
    Trích xuất frame từ video một cách đơn giản, không nhân bản.
    """
    cap = cv2.VideoCapture(video_path)
    count, saved = 0, 0

    # Tạo thư mục đích cho video này nếu chưa có
    os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            # Nếu ảnh nằm ngang, xoay lại thành dọc
            h, w = frame.shape[:2]
            if w > h:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Đặt tên file chuẩn: frame_0000.png, frame_0001.png...
            frame_path = os.path.join(output_folder, f"frame_{saved:04d}.png")
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1
    cap.release()

def process_all_videos(video_root, output_root, max_frames=300):
    labels = ['real', 'fake']

    for label in labels:
        input_dir = os.path.join(video_root, label)
        output_dir_label = os.path.join(output_root, label) # Lưu vào 'real' và 'fake', bỏ hậu tố '_seq'
        
        if not os.path.exists(input_dir):
            print(f"⚠️ Bỏ qua: Không tìm thấy thư mục {input_dir}")
            continue

        videos = [f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        
        for video_file in videos:
            video_path = os.path.join(input_dir, video_file)
            video_name = os.path.splitext(video_file)[0] # Lấy tên video (bỏ đuôi .mp4)
            
            # ĐƯỜNG DẪN MỚI: /dataset_frames/real/tên_video/
            out_video_dir = os.path.join(output_dir_label, video_name)
            
            print(f"🔍 Processing: {video_file} -> {out_video_dir}")
            
            # Cắt thẳng vào thư mục của video đó, KHÔNG DÙNG THƯ MỤC TEMP NỮA
            extract_frames(video_path, out_video_dir, max_frames=max_frames, step=1)
            
        print(f"✅ {label.upper()}: Hoàn tất cắt frame.")

if __name__ == "__main__":
    # 1. Đọc video từ Drive
    input_dir = r"/content/drive/MyDrive/ml_course/project2_face_anti/data/video_goc" 
    
    # 2. Lưu vào ổ CỤC BỘ của Colab trước
    colab_output_dir = r"/content/dataset_frames" 
    
    print("🚀 Bắt đầu quá trình cắt frame siêu nhẹ trên ổ cục bộ Colab...")
    # Không còn cần truyền seq_len và stride vào đây nữa
    process_all_videos(input_dir, colab_output_dir, max_frames=300)
    
    # 3. Nén toàn bộ folder output thành 1 file zip duy nhất
    print("📦 Đang nén dữ liệu thành file zip...")
    zip_path = r"/content/rppg_frames_dataset" 
    shutil.make_archive(zip_path, 'zip', colab_output_dir)
    
    # 4. Copy duy nhất 1 file zip đó lên Google Drive
    print("☁️ Đang chuyển file zip lên Google Drive...")
    drive_dest = r"/content/drive/MyDrive/ml_course/project2_face_anti/data/rppg_frames_dataset.zip"
    shutil.copy(zip_path + ".zip", drive_dest)
    
    print(f"✅ Hoàn tất! File Zip của bạn giờ đây đã nhẹ hơn rất nhiều và được lưu an toàn tại: {drive_dest}")