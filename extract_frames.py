import os
import cv2
import shutil

def extract_frames(video_path, output_folder, max_frames=100, step=1):
    """
    Trích xuất frame từ video (Tối đa 100 frames)
    """
    cap = cv2.VideoCapture(video_path)
    count, saved = 0, 0
    os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            h, w = frame.shape[:2]
            if w > h:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            frame_path = os.path.join(output_folder, f"frame_{saved:04d}.png")
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1
    cap.release()
    return saved # Trả về số lượng đã lưu để kiểm tra

def zip_and_upload(local_dir, drive_dir, batch_num):
    """
    Hàm thực hiện: Nén Zip -> Đẩy lên Drive -> Dọn rác Colab
    """
    print(f"\n📦 Đang đóng gói Batch {batch_num}...")
    zip_filename = f"dataset_batch_{batch_num:03d}"
    zip_local_path = os.path.join("/content", zip_filename)

    # 1. Nén thư mục
    shutil.make_archive(zip_local_path, 'zip', local_dir)

    # 2. Chuyển lên Drive
    drive_dest = os.path.join(drive_dir, f"{zip_filename}.zip")
    print(f"☁️ Đang upload Batch {batch_num} lên Drive: {drive_dest}")
    shutil.copy(f"{zip_local_path}.zip", drive_dest)

    # 3. DỌN DẸP BỘ NHỚ COLAB CHỐNG TRÀN ĐĨA
    print("🧹 Đang dọn dẹp bộ nhớ tạm Colab...")
    os.remove(f"{zip_local_path}.zip")
    shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True) # Tạo lại thư mục rỗng cho batch sau
    
    print(f"✅ Hoàn tất Batch {batch_num}!\n")

def process_and_batch_videos(video_root, colab_temp_root, drive_zip_dir, max_frames=100, batch_size=30):
    labels = ['real', 'fake']
    os.makedirs(drive_zip_dir, exist_ok=True)

    batch_count = 1
    videos_in_current_batch = 0

    for label in labels:
        input_dir = os.path.join(video_root, label)
        if not os.path.exists(input_dir):
            continue

        videos = [f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        
        for video_file in videos:
            video_path = os.path.join(input_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            
            # Lưu tạm trên Colab: /content/dataset_frames/real/video_name/
            out_video_dir = os.path.join(colab_temp_root, label, video_name)
            
            print(f"🔍 Cắt frame: {video_file} -> {out_video_dir}")
            frames_extracted = extract_frames(video_path, out_video_dir, max_frames=max_frames, step=1)
            
            if frames_extracted < max_frames:
                 print(f"   ⚠️ LƯU Ý: Video này ngắn, chỉ trích xuất được {frames_extracted}/{max_frames} frame.")
            
            videos_in_current_batch += 1
            
            # NẾU ĐẠT NGƯỠNG (batch_size) -> NÉN, UPLOAD VÀ XÓA CỤC BỘ
            if videos_in_current_batch >= batch_size:
                zip_and_upload(colab_temp_root, drive_zip_dir, batch_count)
                batch_count += 1
                videos_in_current_batch = 0

    # Xử lý nốt những video còn dư ở batch cuối cùng
    if videos_in_current_batch > 0:
        zip_and_upload(colab_temp_root, drive_zip_dir, batch_count)

if __name__ == "__main__":
    # 1. Thư mục chứa video gốc trên Drive
    input_dir = r"/content/drive/MyDrive/ml_course/project2_face_anti/data/video_goc" 
    
    # 2. Thư mục chứa các file ZIP kết quả trên Drive
    drive_dest_dir = r"/content/drive/MyDrive/ml_course/project2_face_anti/data/rppg_batches"
    
    # 3. Thư mục tạm trên Colab (sẽ liên tục bị xóa và tạo lại)
    colab_temp_dir = r"/content/dataset_frames" 
    
    print("🚀 BẮT ĐẦU TRÍCH XUẤT VÀ CHIA LÔ (BATCHING)...")
    # batch_size=30 nghĩa là cứ xử lý xong 30 video sẽ đẩy lên drive và dọn rác 1 lần
    process_and_batch_videos(input_dir, colab_temp_dir, drive_dest_dir, max_frames=100, batch_size=30)
    
    print("🎉 HOÀN TẤT TOÀN BỘ DỰ ÁN CẮT VIDEO!")