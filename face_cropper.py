import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

class FaceCropper:
    def __init__(self, margin_ratio=0.2, smoothing_factor=0.3):
        """
        Args:
            margin_ratio: Tỷ lệ mở rộng khung mặt (mặc định 20% mỗi cạnh).
            smoothing_factor (alpha): Hệ số làm mượt (0.0 đến 1.0). 
                                      - Nhỏ (0.1) = Khung cắt cực êm nhưng di chuyển trễ.
                                      - Lớn (0.8) = Khung cắt bám cực nhanh nhưng dễ bị rung.
                                      - 0.3 là mức cân bằng tốt nhất cho rPPG.
        """
        self.margin_ratio = margin_ratio
        self.alpha = smoothing_factor
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

    def crop_sequence(self, pil_images):
        """
        Nhận vào một danh sách ảnh PIL (1 sequence),
        trả về danh sách ảnh đã crop với bounding box được làm mượt.
        """
        cropped_images = []
        prev_bbox = None # Lưu trạng thái tọa độ của frame ngay trước đó

        for pil_img in pil_images:
            image_np = np.array(pil_img)
            
            # Đảm bảo ảnh ở định dạng RGB cho MediaPipe
            if len(image_np.shape) == 2: # Ảnh xám
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[2] == 4: # Ảnh RGBA
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

            h, w, _ = image_np.shape
            results = self.face_detection.process(image_np)

            current_bbox = None

            # 1. Dò tìm khuôn mặt
            if results.detections:
                detection = results.detections[0] # Lấy mặt bự/tự tin nhất
                bboxC = detection.location_data.relative_bounding_box
                # Lưu mảng: [x_min, y_min, width, height]
                current_bbox = np.array([bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height])
            
            # 2. Thuật toán làm mượt EMA (Chống rung giật tín hiệu rPPG)
            if current_bbox is not None:
                if prev_bbox is None:
                    prev_bbox = current_bbox # Khởi tạo ở frame đầu tiên
                else:
                    # Công thức EMA: Box_mới = alpha * Box_hiện_tại + (1 - alpha) * Box_cũ
                    prev_bbox = self.alpha * current_bbox + (1 - self.alpha) * prev_bbox
            else:
                # Fallback: Nếu tự nhiên có 1 frame bị mất mặt (do bị che, mờ), xài lại box của frame liền trước
                if prev_bbox is None:
                    prev_bbox = np.array([0.0, 0.0, 1.0, 1.0]) # Lấy full ảnh nếu frame 1 đã xịt

            # 3. Giải mã tọa độ (đã làm mượt) ra pixel thực tế
            rel_x, rel_y, rel_w, rel_h = prev_bbox
            xmin = int(rel_x * w)
            ymin = int(rel_y * h)
            box_w = int(rel_w * w)
            box_h = int(rel_h * h)

            # 4. Tính toán lượng lề (margin) cần mở rộng
            margin_x = int(box_w * self.margin_ratio)
            margin_y = int(box_h * self.margin_ratio)

            # 5. Cắt ảnh (đảm bảo tọa độ không âm hoặc tràn viền)
            new_xmin = max(0, xmin - margin_x)
            new_ymin = max(0, ymin - margin_y)
            new_xmax = min(w, xmin + box_w + margin_x)
            new_ymax = min(h, ymin + box_h + margin_y)

            cropped_np = image_np[new_ymin:new_ymax, new_xmin:new_xmax]
            cropped_images.append(Image.fromarray(cropped_np))

        return cropped_images