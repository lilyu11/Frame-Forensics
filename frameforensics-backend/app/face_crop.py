"""Face cropping and model-input preprocessing for FrameForensics.

This module reproduces, step for step, how the training images were made in
`process_data.py` (`process_crop_face`), so that what the classifier sees at
inference time is the same kind of image it was trained on:

    1. MediaPipe Tasks FaceDetector, BlazeFace short range, run on the RGB frame
    2. the largest detected face (by bounding-box area)
    3. box expanded by CROP_MARGIN (30%) of its width/height on every side,
       clamped to the frame, integer arithmetic exactly as in training
    4. cv2.resize to 224x224 with INTER_AREA (non-square crops are squashed,
       as they were in training; do not "fix" this without retraining)
    5. quality gates: face_conf >= 0.5, Laplacian variance >= 50 (on the crop
       before resizing), bbox_area / frame_area >= 0.001

Model input = BGR->RGB, /255, ImageNet mean/std, NCHW float32.

Keep the constants below in sync with `process_data.py`.
"""
from __future__ import annotations

import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# ---- Must match process_data.py ------------------------------------------------
CROP_MARGIN = 0.30
TARGET_SIZE = (224, 224)
FACE_CONF_THRESHOLD = 0.5
BLUR_THRESHOLD = 50.0
MIN_BBOX_AREA_RATIO = 0.001
DETECTOR_MIN_CONFIDENCE = 0.1   # detector is permissive; the gates above do the filtering

# ---- Must match the training transforms (val_transforms) ----------------------
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
DEFAULT_DETECTOR_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "blaze_face_short_range.tflite"
)

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 in source-frame pixels


@dataclass
class CropResult:
    status: str                       # "ok" | "no_face" | "low_conf" | "blurry" | "small_face" | "empty_crop"
    image: Optional[np.ndarray] = None  # 224x224 BGR uint8 when a crop exists
    box: Optional[Box] = None         # expanded crop box in the source frame
    face_conf: float = 0.0
    blur_score: float = 0.0
    bbox_area_ratio: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def expand_box(bx: int, by: int, bw: int, bh: int, w: int, h: int,
               margin: float = CROP_MARGIN) -> Box:
    """Expand a detector box by `margin` of its size on each side and clamp to the frame.

    Identical to the arithmetic in process_data.process_crop_face.
    """
    mw = int(bw * margin)
    mh = int(bh * margin)
    x1 = max(0, bx - mw)
    y1 = max(0, by - mh)
    x2 = min(w, bx + bw + mw)
    y2 = min(h, by + bh + mh)
    return x1, y1, x2, y2


def ensure_detector_model(path: Path = DEFAULT_DETECTOR_PATH) -> Path:
    """Return the BlazeFace model path, downloading it once if it is missing."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[FaceCropper] Downloading BlazeFace model to {path}")
    try:
        urllib.request.urlretrieve(DETECTOR_URL, str(path))
    except Exception as e:  # pragma: no cover - network dependent
        raise RuntimeError(
            f"Could not download the MediaPipe face detector ({e}). "
            f"Download it manually from {DETECTOR_URL} and place it at {path}."
        ) from e
    return path


def _build_mediapipe_detector(model_path: Path):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=DETECTOR_MIN_CONFIDENCE,
    )
    return vision.FaceDetector.create_from_options(options)


class FaceCropper:
    """Detects the largest face and returns the training-style 224x224 crop.

    `detector` can be injected for tests; it must expose `.detect(mp_image)`.
    The MediaPipe detector is not guaranteed thread-safe, so calls are serialised.
    """

    def __init__(self, model_path: Optional[Path] = None, detector=None,
                 margin: float = CROP_MARGIN):
        self.margin = margin
        self._detector = detector
        self._model_path = Path(model_path) if model_path else DEFAULT_DETECTOR_PATH
        self._lock = threading.Lock()

    def _get_detector(self):
        if self._detector is None:
            self._detector = _build_mediapipe_detector(ensure_detector_model(self._model_path))
        return self._detector

    def crop(self, frame_bgr: np.ndarray) -> CropResult:
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self._lock:
            result = self._get_detector().detect(mp_image)
        if not result.detections:
            return CropResult(status="no_face")

        best = max(result.detections,
                   key=lambda d: d.bounding_box.width * d.bounding_box.height)
        bb = best.bounding_box
        bx, by, bw, bh = bb.origin_x, bb.origin_y, bb.width, bb.height
        face_conf = float(best.categories[0].score)
        bbox_area_ratio = float((bw * bh) / (w * h)) if (w * h) > 0 else 0.0

        box = expand_box(bx, by, bw, bh, w, h, self.margin)
        x1, y1, x2, y2 = box
        face = frame_bgr[y1:y2, x1:x2]
        if face.size == 0:
            return CropResult(status="empty_crop", face_conf=face_conf,
                              bbox_area_ratio=bbox_area_ratio)

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        resized = cv2.resize(face, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        common = dict(image=resized, box=box, face_conf=face_conf,
                      blur_score=blur_score, bbox_area_ratio=bbox_area_ratio)
        # Same order as the training quality filter.
        if face_conf < FACE_CONF_THRESHOLD:
            return CropResult(status="low_conf", **common)
        if blur_score < BLUR_THRESHOLD:
            return CropResult(status="blurry", **common)
        if bbox_area_ratio < MIN_BBOX_AREA_RATIO:
            return CropResult(status="small_face", **common)
        return CropResult(status="ok", **common)

    def close(self):
        det, self._detector = self._detector, None
        if det is not None and hasattr(det, "close"):
            det.close()


def preprocess_crop(crop_bgr_224: np.ndarray) -> np.ndarray:
    """224x224 BGR uint8 crop -> float32 tensor [1, 3, 224, 224] for the ONNX model.

    No resize happens here: the crop was already resized with INTER_AREA, which is what
    produced the training images (so the training-time Resize((224,224)) was a no-op).
    """
    if crop_bgr_224.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
        raise ValueError(f"expected a {TARGET_SIZE[0]}x{TARGET_SIZE[1]} crop, got {crop_bgr_224.shape}")
    rgb = cv2.cvtColor(crop_bgr_224, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
