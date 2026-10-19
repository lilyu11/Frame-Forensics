import json
import os
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import imageio_ffmpeg
import numpy as np
import onnxruntime as ort

from face_crop import FaceCropper, preprocess_crop

# CONFIG
BACKEND_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get(
    "FF_MODEL_PATH", BACKEND_ROOT / "models" / "frameforensics_mnv3s_fp32.onnx"))

# Class order used in training (FaceDataset: REAL -> 1, everything else -> 0).
IDX_FAKE, IDX_REAL = 0, 1

FAKE_THRESHOLD = 0.5       # smoothed P(fake) above this => FAKE
EMA_ALPHA = 0.6            # weight of the newest observation in the EMA
SAMPLE_INTERVAL_S = 0.5    # analyse ~2 frames per second
MAX_MISSES = 3             # consecutive sampled frames without a usable face before the verdict is dropped

GREEN, RED, GRAY = (0, 255, 0), (0, 0, 255), (160, 160, 160)   # BGR


# ONNX RUNTIME ENGINE
class ONNXDeepfakeEngine:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}. Run export_to_onnx.ipynb first "
                f"or set FF_MODEL_PATH.")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Refuse to run if the model was exported with a different class order.
        meta = self.session.get_modelmeta().custom_metadata_map
        if "class_index" in meta:
            idx = json.loads(meta["class_index"])
            if idx != {"FAKE": IDX_FAKE, "REAL": IDX_REAL}:
                raise ValueError(f"Model class_index {idx} does not match IDX_FAKE/IDX_REAL in app.py")
        print(f"[ONNXEngine] Loaded: {self.model_path}")

    def predict_fake_prob(self, crop_bgr_224: np.ndarray) -> float:
        """Probability that the face crop is FAKE (softmax over the two logits)."""
        x = preprocess_crop(crop_bgr_224)
        logits = self.session.run([self.output_name], {self.input_name: x})[0][0]
        e = np.exp(logits - np.max(logits))
        probs = e / e.sum()
        return float(probs[IDX_FAKE])


@lru_cache(maxsize=1)
def get_engine() -> ONNXDeepfakeEngine:
    return ONNXDeepfakeEngine()


@lru_cache(maxsize=1)
def get_cropper() -> FaceCropper:
    return FaceCropper()


# TEMPORAL SMOOTHING (EMA)
class TemporalSmoother:
    """Smooths P(fake) over time so the label does not flicker on video/webcam."""

    def __init__(self, alpha=EMA_ALPHA):
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


# PER-STREAM ANALYSER (one per video or per webcam session)
@dataclass
class Verdict:
    label: str = "ANALYZING..."
    confidence: float = 0.0                 # percent, only meaningful for FAKE/REAL
    color: Tuple[int, int, int] = GRAY
    box: Optional[Tuple[int, int, int, int]] = None

    def text(self) -> str:
        if self.label.startswith(("FAKE", "REAL")):
            return f"{self.label} ({self.confidence:.1f}%)"
        return self.label


class StreamAnalyzer:
    def __init__(self):
        self.smoother = TemporalSmoother()
        self.misses = 0
        self.verdict = Verdict()
        self.last_sample_t = 0.0

    def analyze(self, frame_bgr: np.ndarray) -> Verdict:
        crop = get_cropper().crop(frame_bgr)

        if not crop.ok:
            self.misses += 1
            if self.misses >= MAX_MISSES:
                self.smoother.reset()
                label = "LOW QUALITY" if crop.image is not None else "NO FACE"
                self.verdict = Verdict(label=label, color=GRAY, box=crop.box)
            return self.verdict

        self.misses = 0
        score = self.smoother.update(get_engine().predict_fake_prob(crop.image))
        if score > FAKE_THRESHOLD:
            self.verdict = Verdict("FAKE (DEEPFAKE)", score * 100, RED, crop.box)
        else:
            self.verdict = Verdict("REAL", (1.0 - score) * 100, GREEN, crop.box)
        return self.verdict

    def due(self, now: float) -> bool:
        if now - self.last_sample_t >= SAMPLE_INTERVAL_S:
            self.last_sample_t = now
            return True
        return False


def draw_overlay(img_bgr: np.ndarray, verdict: Verdict) -> np.ndarray:
    if verdict.box is not None:
        x1, y1, x2, y2 = verdict.box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), verdict.color, 2)
    scale = max(0.6, img_bgr.shape[1] / 1000.0)
    cv2.putText(img_bgr, f"Status: {verdict.text()}", (20, int(40 * scale) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, scale, verdict.color, 2, cv2.LINE_AA)
    return img_bgr


# VIDEO PIPELINE (TAB 1)
def process_video_file(video_path):
    if not video_path:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Không mở được video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_file.close()
    # H.264 via the ffmpeg binary bundled with imageio-ffmpeg: OpenCV's mp4v output does not
    # play in browsers, and OpenCV's own H.264 support depends on the platform.
    out = imageio_ffmpeg.write_frames(
        out_file.name, (width, height), fps=fps, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", macro_block_size=2, quality=7, ffmpeg_log_level="error")
    out.send(None)

    analyzer = StreamAnalyzer()
    frame_interval = max(1, round(fps * SAMPLE_INTERVAL_S))   # ~2 analysed frames per second
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                analyzer.analyze(frame)
            annotated = draw_overlay(frame, analyzer.verdict)
            out.send(np.ascontiguousarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)))
            frame_idx += 1
    finally:
        cap.release()
        out.close()
    return out_file.name


# WEBCAM PIPELINE (TAB 2)
def process_webcam_frame(frame_rgb, state):
    """Gradio gives RGB frames; everything below works in BGR like OpenCV and the training crops."""
    if frame_rgb is None:
        return None, state
    if state is None:
        state = StreamAnalyzer()

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if state.due(time.monotonic()):
        state.analyze(frame_bgr)
    annotated = draw_overlay(frame_bgr, state.verdict)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), state


# GRADIO UI
with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ Real-time Deepfake Detection MVP (CPU-Optimized)")
    gr.Markdown(
        "Hệ thống phát hiện Deepfake tối ưu cho CPU: MediaPipe cắt khuôn mặt (giống lúc huấn luyện), "
        "ONNX Runtime, Frame Sampling 2 FPS và Temporal Smoothing.")

    with gr.Tabs():
        with gr.TabItem("📁 Phân tích Video"):
            with gr.Row():
                video_input = gr.Video(label="Tải lên video cần kiểm tra")
                video_output = gr.Video(label="Kết quả xử lý (Annotated Video)")
            btn_run_video = gr.Button("🚀 Chạy Phân Tích", variant="primary")
            btn_run_video.click(fn=process_video_file, inputs=video_input, outputs=video_output)

        with gr.TabItem("📷 Live Webcam Stream"):
            gr.Markdown("Hệ thống tự động lấy mẫu 2 FPS để không giật lag trên CPU. Cần thấy rõ khuôn mặt, đủ sáng.")
            webcam_state = gr.State(None)
            webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam trực tiếp")
            webcam_output = gr.Image(label="Kết quả nhận diện Realtime")
            webcam_input.stream(fn=process_webcam_frame,
                                inputs=[webcam_input, webcam_state],
                                outputs=[webcam_output, webcam_state])

if __name__ == "__main__":
    get_engine()    # fail fast with a clear message if the model file is missing
    port = int(os.environ.get("PORT", os.environ.get("FF_PORT", 7860)))
    demo.launch(server_name="0.0.0.0", server_port=port,
                share=False, inbrowser=False, theme=gr.themes.Soft())
