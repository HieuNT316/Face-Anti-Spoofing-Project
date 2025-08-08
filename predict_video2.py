import os
import threading
import torch
import numpy as np
import cv2
from PIL import Image, ImageTk
from torchvision import transforms
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import time

# Thêm import cho biểu đồ
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm

# =========================
# CONFIG
# =========================
# The code assumes these models and paths exist.
# Make sure to adjust these paths if needed.
SEQ_LEN = 100
ALPHA = 0.8
THRESHOLD = 0.5356

RPPG_MIN, RPPG_MAX = 0.0034, 0.5022
DEPTH_MIN, DEPTH_MAX = 3388.8628, 8643.7754

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Models
# =========================
# These imports and model loading steps are assumed to work based on the user's setup.
try:
    from models.unet_depth_cnn import UNetDepthCNN
    from models.rppg_rnn import RPPG_RNN

    print("🔍 Loading models...")
    depth_model = UNetDepthCNN().to(device)
    depth_model.load_state_dict(torch.load("unet_depth_epoch20.pth", map_location=device))
    depth_model.eval()

    rppg_model = RPPG_RNN().to(device)
    rppg_model.load_state_dict(torch.load("rppg_epoch5.pth", map_location=device))
    rppg_model.eval()
    print("✅ Models loaded successfully.")

except Exception as e:
    print(f"❌ Error loading models: {e}. The app will still run, but prediction will fail.")
    depth_model = None
    rppg_model = None


# =========================
# Frame Extraction
# =========================
def extract_frames(video_path, num_frames=SEQ_LEN, progress_callback=None):
    """
    Extracts frames from a video and updates a progress bar.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use a minimum of 1 frame, but not more than total frames
    actual_num_frames = min(num_frames, total_frames_in_video)
    indices = np.linspace(0, total_frames_in_video - 1, actual_num_frames, dtype=int)
    
    frames, raw_imgs = [], []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_imgs.append(Image.fromarray(frame_rgb))
        img_tensor = transform(Image.fromarray(frame_rgb))
        frames.append(img_tensor)
        
        # Call the progress callback to update the UI
        if progress_callback:
            progress_callback(i + 1, actual_num_frames)
            
    cap.release()
    return torch.stack(frames).unsqueeze(0).to(device), raw_imgs

# =========================
# Prediction
# =========================
@torch.no_grad()
def predict_score(seq_tensor):
    """
    Performs the prediction using the loaded models.
    """
    if depth_model is None or rppg_model is None:
        raise RuntimeError("Models are not loaded. Cannot perform prediction.")

    # RPPG score calculation
    rppg_feat = rppg_model(seq_tensor)
    rppg_score = torch.norm(rppg_feat, p=2) ** 2

    # Depth score calculation
    first_frame = seq_tensor[0, 0]  # [3, H, W]
    depth_map = depth_model(first_frame.unsqueeze(0))
    depth_score = torch.norm(depth_map, p=2) ** 2

    # Normalization and total score
    rppg_norm = (rppg_score.item() - RPPG_MIN) / (RPPG_MAX - RPPG_MIN + 1e-8)
    depth_norm = (depth_score.item() - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN + 1e-8)
    total_score = rppg_norm + ALPHA * depth_norm

    return {
        "total_score": total_score,
        "rppg_score": rppg_score.item(),
        "depth_score": depth_score.item(),
        "rppg_norm": rppg_norm,
        "depth_norm": depth_norm
    }

# =========================
# GUI App (ttkbootstrap)
# =========================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("🛡️ Face Anti-Spoofing Detector")
        self.root.geometry("1600x900")  # Tăng kích thước cửa sổ
        self.root.resizable(True, True)  # Cho phép thay đổi kích thước cửa sổ
        
        self.video_path = ""
        self.raw_imgs = []

        self.build_ui()

    def build_ui(self):
        """Builds the main user interface."""
        # Main title
        ttk.Label(self.root, text="🛡️ Face Anti-Spoofing Detection", font=("Helvetica", 36, "bold"), bootstyle=PRIMARY).pack(pady=(20, 10))
        
        # Container for file selection and buttons
        input_frame = ttk.Frame(self.root, padding=10)
        input_frame.pack(pady=10, fill=X)
        
        # File path entry
        self.entry = ttk.Entry(input_frame, width=80, font=("Helvetica", 14))
        self.entry.pack(side=LEFT, padx=10, ipady=6, fill=X, expand=True)

        # Buttons
        ttk.Button(input_frame, text="Browse", command=self.select_file, bootstyle=INFO, width=10, padding=10).pack(side=LEFT, padx=5)
        ttk.Button(input_frame, text="Predict", command=self.run_prediction, bootstyle=SUCCESS, width=10, padding=10).pack(side=LEFT, padx=5)
        ttk.Button(input_frame, text="Copy Link", command=self.copy_link, bootstyle=LIGHT, width=10, padding=10).pack(side=LEFT, padx=5)

        # Progress bar
        self.progress_bar = ttk.Progressbar(self.root, mode='determinate', length=1200, bootstyle=SUCCESS)
        self.progress_bar.pack(pady=10)

        # Container for preview and results
        main_content_frame = ttk.Frame(self.root)
        main_content_frame.pack(pady=10, padx=20, fill=BOTH, expand=True)
        
        # Video Preview Section with a nice border
        preview_container = ttk.LabelFrame(main_content_frame, text="Video Preview", padding=10, bootstyle=SECONDARY)
        preview_container.pack(side=LEFT, padx=(0, 20), fill=BOTH, expand=True)
        
        preview_border_frame = ttk.Frame(preview_container, borderwidth=3, relief="groove")
        preview_border_frame.pack(fill=BOTH, expand=True)
        
        self.preview_label = ttk.Label(preview_border_frame, text="Select a video file to begin", font=("Helvetica", 16), bootstyle=LIGHT)
        self.preview_label.pack(fill=BOTH, expand=True)
        
        try:
            placeholder_img = Image.new("RGB", (800, 600), "gray")
            self.placeholder_photo = ImageTk.PhotoImage(placeholder_img)
            self.preview_label.config(image=self.placeholder_photo)
            self.preview_label.image = self.placeholder_photo
        except Exception as e:
            print(f"Error creating placeholder image: {e}")
        
        # Detection Result and Chart Section
        result_chart_container = ttk.Frame(main_content_frame)
        result_chart_container.pack(side=RIGHT, fill=BOTH, expand=True)

        result_container = ttk.LabelFrame(result_chart_container, text="Detection Result", padding=20, bootstyle=SECONDARY)
        result_container.pack(pady=(0, 10), fill=X)
        
        self.status_label = ttk.Label(result_container, text="Status: Ready", font=("Helvetica", 18), bootstyle=PRIMARY)
        self.status_label.pack(pady=10)
        
        self.result_label = ttk.Label(result_container, text="Result: Awaiting prediction...", font=("Helvetica", 28, "bold"))
        self.result_label.pack(pady=20)
        
        self.detail_label = ttk.Label(result_container, text="", font=("Helvetica", 14), wraplength=500, justify=LEFT)
        self.detail_label.pack(pady=10, fill=BOTH, expand=True)

        # Chart Section
        chart_container = ttk.LabelFrame(result_chart_container, text="Score Visualization", padding=10, bootstyle=SECONDARY)
        chart_container.pack(fill=BOTH, expand=True)
        self.chart_frame = ttk.Frame(chart_container)
        self.chart_frame.pack(fill=BOTH, expand=True)
        self.create_chart(self.chart_frame, {"rppg": 0, "depth": 0})
        
        # Removed Customization Section
        # This section with alpha and threshold sliders has been removed as per request.

    def create_chart(self, parent_frame, scores):
        """Creates and embeds a matplotlib bar chart."""
        try:
            # Clear existing chart if any
            for widget in parent_frame.winfo_children():
                widget.destroy()

            fig = plt.Figure(figsize=(5, 3), dpi=100)
            ax = fig.add_subplot(111)
            
            labels = ['RPPG Score', 'Depth Score']
            values = [scores["rppg"], scores["depth"]]
            colors = ['#17a2b8', '#fd7e14']
            
            bars = ax.bar(labels, values, color=colors)
            ax.set_ylabel('Normalized Score')
            ax.set_title('Score')
            ax.set_ylim(0, 2.0)
            
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=parent_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=BOTH, expand=True)
        except Exception as e:
            print(f"Error creating chart: {e}")

    def select_file(self):
        """Opens a file dialog to select a video file."""
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if path:
            self.entry.delete(0, END)
            self.entry.insert(0, path)
            self.video_path = path
            self.reset_ui()
    
    def copy_link(self):
        """Copies the video path from the entry widget to the clipboard."""
        path = self.entry.get()
        if path:
            self.root.clipboard_clear()
            self.root.clipboard_append(path)
            self.status_label.config(text="Status: Link copied to clipboard!", bootstyle=INFO)
        else:
            messagebox.showinfo("Thông báo", "Vui lòng chọn một file video trước.")

    def reset_ui(self):
        """Resets the UI elements to their initial state."""
        self.result_label.config(text="Result: Awaiting prediction...", bootstyle=SECONDARY)
        self.detail_label.config(text="")
        self.progress_bar.config(mode='determinate', value=0)
        self.status_label.config(text="Status: Ready", bootstyle=PRIMARY)
        self.create_chart(self.chart_frame, {"rppg": 0, "depth": 0})
        try:
            self.preview_label.config(image=self.placeholder_photo, text="")
        except Exception as e:
            print(f"Error restoring placeholder image: {e}")

    def update_progress(self, current_frame, total_frames):
        """Updates the progress bar and status label."""
        self.progress_bar['value'] = (current_frame / total_frames) * 100
        self.status_label.config(text=f"Status: Extracting frames... ({current_frame}/{total_frames})")
        self.root.update_idletasks()

    def run_prediction(self):
        """Starts the prediction process in a separate thread."""
        video_path = self.entry.get()
        if not os.path.exists(video_path):
            messagebox.showerror("Error", "Video path invalid or file not found.")
            return

        self.status_label.config(text="Status: Starting prediction...", bootstyle=INFO)
        self.result_label.config(text="Result: In progress...", bootstyle=SECONDARY)
        self.detail_label.config(text="Extracting frames and processing video...")
        
        threading.Thread(target=self._run, args=(video_path,), daemon=True).start()

    def _run(self, video_path):
        """The main logic for prediction, run in a separate thread."""
        try:
            # Stage 1: Extract frames with progress
            self.progress_bar.config(mode='determinate', maximum=100)
            seq_tensor, self.raw_imgs = extract_frames(video_path, progress_callback=self.update_progress)
            
            # Stage 2: Show video preview
            self.status_label.config(text="Status: Playing preview...", bootstyle=INFO)
            for frame in self.raw_imgs[:min(len(self.raw_imgs), 30)]:
                frame = frame.resize((800, 600), Image.Resampling.LANCZOS)
                img = ImageTk.PhotoImage(frame)
                self.preview_label.config(image=img, text="")
                self.preview_label.image = img
                time.sleep(0.05)

            # Stage 3: Predict score
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            self.status_label.config(text="Status: Analyzing video...", bootstyle=INFO)
            
            # Use the global ALPHA value
            result = predict_score(seq_tensor)

            # Stage 4: Display results
            self.progress_bar.stop()
            self.progress_bar.config(value=100)
            
            # Use the global THRESHOLD value
            verdict = "✅ REAL" if result['total_score'] > THRESHOLD else "❌ FAKE"
            color = SUCCESS if verdict == "✅ REAL" else DANGER

            self.status_label.config(text="Status: Done!", bootstyle=SUCCESS)
            self.result_label.config(text=f"Result: {verdict}", bootstyle=color)
            self.detail_label.config(
                text=f"Score: {result['total_score']:.4f}\n"
                     f"RPPG Score: {result['rppg_score']:.2f} | Depth Score: {result['depth_score']:.2f}\n"
                     f"Normalized RPPG: {result['rppg_norm']:.4f} | Normalized Depth: {result['depth_norm']:.4f}"
            )
            self.create_chart(self.chart_frame, {"rppg": result['rppg_norm'], "depth": result['depth_norm']})

        except Exception as e:
            self.progress_bar.stop()
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Status: Error", bootstyle=DANGER)
            print(f"Prediction failed with error: {e}")

# =========================
# Run app
# =========================
if __name__ == "__main__":
    # flatly superhero
    app = ttk.Window(themename="flatly")
    App(app)
    app.mainloop()
