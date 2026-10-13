import cv2
import numpy as np
import onnxruntime as ort
import gradio as gr
import time

# ONNX RUNTIME ENGINE
class ONNXDeepfakeEngine:
    # Khởi tạo engine ONNX Runtime (hiện chưa có)
    def __init__(self, model_path="engine_path_chưa_có"):
        self.model_path = model_path
        
        # Khởi tạo ONNX Runtime Session
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        
        # Lấy tên các node input/output của model ONNX
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"[ONNXEngine] Đã nạp thành công model từ: {self.model_path}")

    def preprocess(self, face_img):
        """
        Tiền xử lý ảnh khuôn mặt trước khi đưa vào mô hình:
        1. Resize về kích thước chuẩn (224x224)
        2. Chuyển đổi không gian màu BGR sang RGB
        3. Chuẩn hóa giá trị pixel về đoạn [0, 1] và áp dụng Mean/Std chuẩn ImageNet
        4. Chuyển chiều từ (H, W, C) sang (C, H, W) và thêm batch dimension (1, 3, 224, 224)
        """
        img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, face_img):
        """
        Thực hiện inference trực tiếp qua mô hình ONNX:
        - Đầu vào: Ảnh khuôn mặt (numpy array BGR)
        - Đầu ra: (label: str, confidence: float)
        """
        input_tensor = self.preprocess(face_img)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0][0]  # Vector logits [Real_score, Fake_score]
        
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        
        if fake_prob > real_prob:
            label = "FAKE"
            confidence = fake_prob
        else:
            label = "REAL"
            confidence = real_prob
            
        return label, confidence

# Khởi tạo engine với đường dẫn model thực tế (chưa có)
engine = ONNXDeepfakeEngine(model_path="engine_path_chưa_có")


# CƠ CHẾ TEMPORAL SMOOTHING (EMA)
class TemporalSmoother:
    """
    Lọc mượt xác suất dự đoán qua thời gian để tránh nhấp nháy nhãn trên video/webcam.
    """
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.current_score = None

    def update(self, new_score):
        if self.current_score is None:
            self.current_score = new_score
        else:
            self.current_score = self.alpha * new_score + (1 - self.alpha) * self.current_score
        return self.current_score

    def reset(self):
        self.current_score = None


# PIPELINE XỬ LÝ VIDEO (TAB 1)
def process_video_file(video_path):
    if not video_path:
        return None

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    smoother = TemporalSmoother(alpha=0.6)
    frame_interval = max(1, int(fps * 0.5))  # Lấy mẫu mỗi ~500ms (2 FPS)
    
    last_label = "PROCESSING..."
    last_conf = 0.0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Sampling theo chu kỳ 2 FPS
        if frame_idx % frame_interval == 0:
            # [Lưu ý thực tế]: Có thể tích hợp MediaPipe Face Detection ở đây để crop mặt trước, 
            # hoặc truyền trực tiếp frame nếu video đã được cắt mặt sẵn.
            raw_label, raw_conf = engine.predict(frame)
            
            score_val = raw_conf if raw_label == "FAKE" else (1.0 - raw_conf)
            smoothed_score = smoother.update(score_val)
            
            if smoothed_score > 0.5:
                last_label = "FAKE (DEEPFAKE)"
                last_conf = smoothed_score * 100
                color = (0, 0, 255)
            else:
                last_label = "REAL"
                last_conf = (1.0 - smoothed_score) * 100
                color = (0, 255, 0)

        cv2.putText(frame, f"Status: {last_label} ({last_conf:.1f}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return output_path



# PIPELINE WEBCAM REALTIME (TAB 2)
smoother_webcam = TemporalSmoother(alpha=0.6)
last_inference_time = 0
cached_label = "AWAITING..."
cached_conf = 0.0

def process_webcam_frame(frame):
    global last_inference_time, cached_label, cached_conf
    if frame is None:
        return None

    current_time = time.time()
    # Sampling theo chu kỳ 2 FPS
    if current_time - last_inference_time >= 0.5:
        raw_label, raw_conf = engine.predict(frame)
        score_val = raw_conf if raw_label == "FAKE" else (1.0 - raw_conf)
        smoothed_score = smoother_webcam.update(score_val)

        if smoothed_score > 0.5:
            cached_label = "FAKE (DEEPFAKE)"
            cached_conf = smoothed_score * 100
        else:
            cached_label = "REAL"
            cached_conf = (1.0 - smoothed_score) * 100
            
        last_inference_time = current_time

    color = (0, 0, 255) if "FAKE" in cached_label else (0, 255, 0)
    cv2.putText(frame, f"Realtime CPU: {cached_label} ({cached_conf:.1f}%)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    
    return frame



# GIAO DIỆN GRADIO UI (WEB APP)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Real-time Deepfake Detection MVP (CPU-Optimized)")
    gr.Markdown("Hệ thống kiểm chứng nhận diện Deepfake thời gian thực tối ưu cho CPU sử dụng ONNX Runtime, Frame Sampling & Temporal Smoothing.")

    with gr.Tabs():
        with gr.TabItem("📁 Phân tích Video"):
            with gr.Row():
                video_input = gr.Video(label="Tải lên video cần kiểm tra")
                video_output = gr.Video(label="Kết quả xử lý (Annotated Video)")
            btn_run_video = gr.Button("🚀 Chạy Phân Tích", variant="primary")
            btn_run_video.click(fn=process_video_file, inputs=video_input, outputs=video_output)

        with gr.TabItem("📷 Live Webcam Stream"):
            gr.Markdown("Hệ thống tự động lấy mẫu 2 FPS kết hợp ONNX Runtime INT8 để đảm bảo không giật lag trên CPU.")
            webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam trực tiếp")
            webcam_output = gr.Image(label="Kết quả nhận diện Realtime")
            webcam_input.stream(fn=process_webcam_frame, inputs=webcam_input, outputs=webcam_output)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)