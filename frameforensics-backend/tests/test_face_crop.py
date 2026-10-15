"""Checks that face_crop.py reproduces the training crop in process_data.py.

Run from frameforensics-backend/:  python -m pytest tests -q
No network or detector model is needed: the detector is replaced by a stub that
returns fixed boxes, so only the geometry / resizing / gating logic is tested.
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import face_crop as fc            # noqa: E402
import process_data as pd_mod     # noqa: E402  (training pipeline)


class StubDetector:
    def __init__(self, boxes):    # boxes: list of (x, y, w, h, score)
        self.boxes = boxes

    def detect(self, _mp_image):
        dets = [NS(bounding_box=NS(origin_x=x, origin_y=y, width=w, height=h),
                   categories=[NS(score=s)]) for x, y, w, h, s in self.boxes]
        return NS(detections=dets)


def textured_frame(h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)   # high Laplacian variance


@pytest.mark.parametrize("boxes", [
    [(200, 100, 160, 180, 0.97)],
    [(0, 0, 90, 90, 0.80)],                                   # touches top-left: margin must clamp
    [(560, 400, 120, 120, 0.90)],                             # runs past bottom-right: must clamp
    [(50, 50, 40, 40, 0.99), (300, 120, 200, 220, 0.85)],     # two faces: the larger one wins
])
def test_matches_training_crop(boxes):
    frame = textured_frame()
    pd_mod.face_detector = StubDetector(boxes)                # used by process_crop_face
    ref_img, ref_conf, ref_blur, ref_ratio = pd_mod.process_crop_face(frame, margin=pd_mod.CROP_MARGIN)

    res = fc.FaceCropper(detector=StubDetector(boxes)).crop(frame)
    assert res.status == "ok"
    assert np.array_equal(res.image, ref_img)                 # bit-identical pixels
    assert res.face_conf == pytest.approx(ref_conf)
    assert res.blur_score == pytest.approx(ref_blur)
    assert res.bbox_area_ratio == pytest.approx(ref_ratio)


def test_constants_in_sync_with_training():
    assert fc.CROP_MARGIN == pd_mod.CROP_MARGIN
    assert fc.TARGET_SIZE == pd_mod.TARGET_SIZE
    assert fc.FACE_CONF_THRESHOLD == pd_mod.FACE_CONF_THRESHOLD
    assert fc.BLUR_THRESHOLD == pd_mod.BLUR_THRESHOLD
    assert fc.MIN_BBOX_AREA_RATIO == pd_mod.MIN_BBOX_AREA_RATIO


def test_no_face():
    res = fc.FaceCropper(detector=StubDetector([])).crop(textured_frame())
    assert res.status == "no_face" and res.image is None


def test_quality_gates():
    frame = textured_frame()
    assert fc.FaceCropper(detector=StubDetector([(200, 100, 160, 180, 0.30)])).crop(frame).status == "low_conf"
    flat = np.full((480, 640, 3), 128, np.uint8)               # Laplacian variance 0
    assert fc.FaceCropper(detector=StubDetector([(200, 100, 160, 180, 0.95)])).crop(flat).status == "blurry"
    assert fc.FaceCropper(detector=StubDetector([(10, 10, 4, 4, 0.95)])).crop(textured_frame(2000, 2000)).status == "small_face"


def test_preprocess_matches_training_normalisation():
    crop = textured_frame(224, 224, seed=3)                    # BGR uint8, already 224x224
    x = fc.preprocess_crop(crop)
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    rgb = crop[..., ::-1].astype(np.float64) / 255.0           # independent float64 reference
    ref = ((rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)
    assert np.abs(x[0] - ref).max() < 1e-5


def test_preprocess_rejects_wrong_size():
    with pytest.raises(ValueError):
        fc.preprocess_crop(np.zeros((200, 224, 3), np.uint8))
