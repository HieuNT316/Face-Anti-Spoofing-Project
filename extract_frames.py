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

def split_to_sequences(image_folder, output_folder, seq_len=100):
    """
    Chia mảng ảnh thành các sequence liên tiếp.
    Ví dụ: Lấy 300 ảnh chia thành 3 thư mục seq_0001, seq_0002, seq_0003 (Mỗi thư mục 100 ảnh)
    """
    images = sorted(glob(os.path.join(image_folder, "*.png")))
    existing = glob(os.path.join(output_folder, "seq_*"))
    count = len(existing)

    for i in range(0, len(images) - seq_len + 1, seq_len):
        seq_imgs = images[i:i + seq_len]
        seq_folder = os.path.join(output_folder, f"seq_{count:04d}")
        os.makedirs(seq_folder, exist_ok=True)
        
        for j, img_path in enumerate(seq_imgs):
            new_path = os.path.join(seq_folder, f"frame_{j + 1:04d}.png")
            shutil.copy(img_path, new_path)
        count += 1

    return count - len(existing)

def process_all_videos(video_root, output_root, seq_len=100, max_frames=300):
    temp_dir = "temp_frames"
    
    # SỬA LẠI: Thêm cả 'fake' để hệ thống xử lý cả 2 class
    labels = ['real', 'fake']

    for label in labels:
        input_dir = os.path.join(video_root, label)
        output_dir_label = os.path.join(output_root, f"{label}_seq")
        os.makedirs(output_dir_label, exist_ok=True)

        if not os.path.exists(input_dir):
            print(f"⚠️ Không tìm thấy thư mục: {input_dir}")
            continue

        videos = [f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        seq_total = 0

        for video_file in videos:
            video_path = os.path.join(input_dir, video_file)

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)

            print(f"🔍 Đang xử lý: {video_file} (Class: {label.upper()})")
            
            # Khai thác tối đa max_frames (VD: 300), không bỏ cóc (step=1)
            extract_frames(video_path, temp_dir, max_frames=max_frames, step=1)
            
            # Cắt thành các sequence 100 ảnh
            added = split_to_sequences(temp_dir, output_dir_label, seq_len=seq_len)
            
            if added > 0:
                print(f"   ➕ Thêm được {added} sequences (Mỗi seq {seq_len} frames)")
            else:
                print(f"   ❌ Video quá ngắn, không đủ {seq_len} frames. Đã bỏ qua.")
                
            seq_total += added

        print(f"✅ {label.upper()}: Tổng cộng {seq_total} sequences đã tạo.")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Cập nhật đường dẫn của bạn ở đây
    input_dir = r"D:\ml_course\project2_face_anti\data\video_goc" # Thư mục chứa video (.mp4)
    output_dir = r"D:\ml_course\project2_face_anti\data\frames"  # Thư mục lưu frame
    
    # Chạy trích xuất: Mỗi sequence 100 frame, lấy tối đa 300 frame (3 sequence) mỗi video
    process_all_videos(input_dir, output_dir, seq_len=100, max_frames=300)
    print("✅ Xử lý hoàn tất. Bạn có thể nén thư mục 'frames' lại và đưa lên Google Drive!")