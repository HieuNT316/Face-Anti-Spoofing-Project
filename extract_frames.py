import os
import cv2
import shutil
from glob import glob

def extract_frames(video_path, output_folder, max_frames=300, step=1):
    """
    Trích xuất frame từ video.
    Đã đổi step=1 để lấy ảnh LIÊN TỤC, không làm đứt gãy tín hiệu rPPG.
    """
    cap = cv2.VideoCapture(video_path)
    count, saved = 0, 0

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            # Nếu ảnh nằm ngang, xoay lại thành dọc
            h, w = frame.shape[:2]
            if w > h:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            os.makedirs(output_folder, exist_ok=True)
            frame_path = os.path.join(output_folder, f"{saved:04d}.png") # Tăng lên 4 số để chứa đủ 100+ frame
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1
    cap.release()

def split_to_sequences_sliding_window(image_folder, output_folder, seq_len=100, stride=50):
    """
    Chia mảng ảnh thành các sequence sử dụng kỹ thuật Sliding Window.
    stride: Bước nhảy của cửa sổ. 
    Ví dụ: seq_len=100, stride=50 -> Seq1: 0-99, Seq2: 50-149, Seq3: 100-199...
    """
    images = sorted(glob(os.path.join(image_folder, "*.png")))
    
    # Đếm số lượng thư mục seq đã có sẵn để đặt tên tiếp nối
    existing = glob(os.path.join(output_folder, "seq_*"))
    count = len(existing)
    
    added_count = 0
    
    # Vòng lặp trượt cửa sổ
    # range(start, stop, step) -> step ở đây chính là stride
    for i in range(0, len(images) - seq_len + 1, stride):
        seq_imgs = images[i : i + seq_len]
        
        # Tạo thư mục cho sequence mới
        seq_folder = os.path.join(output_folder, f"seq_{count:05d}") # Tăng lên 5 số cho thoải mái
        os.makedirs(seq_folder, exist_ok=True)
        
        for j, img_path in enumerate(seq_imgs):
            # Lưu frame vào folder mới, đánh số từ 0001 đến seq_len
            new_path = os.path.join(seq_folder, f"frame_{j + 1:04d}.png")
            shutil.copy(img_path, new_path)
            
        count += 1
        added_count += 1

    return added_count

def process_all_videos(video_root, output_root, seq_len=100, stride=50, max_frames=300):
    temp_dir = "temp_frames"
    labels = ['real', 'fake']

    for label in labels:
        input_dir = os.path.join(video_root, label)
        output_dir_label = os.path.join(output_root, f"{label}_seq")
        os.makedirs(output_dir_label, exist_ok=True)

        if not os.path.exists(input_dir):
            continue

        videos = [f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        seq_total = 0

        for video_file in videos:
            video_path = os.path.join(input_dir, video_file)
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)

            print(f"🔍 Processing: {video_file} | Class: {label.upper()}")
            
            extract_frames(video_path, temp_dir, max_frames=max_frames, step=1)
            
            # GỌI HÀM SLIDING WINDOW Ở ĐÂY
            added = split_to_sequences_sliding_window(temp_dir, output_dir_label, seq_len=seq_len, stride=stride)
            
            if added > 0:
                print(f"   ➕ Added {added} sequences (Overlap stride: {stride})")
            seq_total += added

        print(f"✅ {label.upper()}: Total {seq_total} sequences created.")

    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

if __name__ == "__main__":
    input_dir = r"D:\ml_course\project2_face_anti\data\video_goc"
    output_dir = r"D:\ml_course\project2_face_anti\data\frames"
    
    # THAY ĐỔI THÔNG SỐ Ở ĐÂY:
    # seq_len=100, stride=50 nghĩa là mỗi sequence gối đầu lên nhau 50%
    # Với max_frames=300, bạn sẽ thu được 5 sequence thay vì 3 như trước.
    # (0-100, 50-150, 100-200, 150-250, 200-300)
    process_all_videos(input_dir, output_dir, seq_len=100, stride=50, max_frames=300)