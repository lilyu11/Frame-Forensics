import os
import urllib.request

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Cấu hình đường dẫn
BASE_ARCHIVE_DIR = Path(r"D:\PROJECT FRAME FORENSICS\frameforensics-backend\archive\FaceForensics++_C23")
OUTPUT_BASE_DIR = Path(r"D:\PROJECT FRAME FORENSICS\frameforensics-backend\processed_data")

# Cấu hình danh sách thư mục dữ liệu (để dễ mở rộng thêm các loại deepfake khác sau này)
DATASET_CONFIG = {
	"original": "REAL",
	"DeepFakeDetection": "FAKE",
	# "Deepfakes": "FAKE",
	# "Face2Face": "FAKE",
	# "FaceShifter": "FAKE",
	# "FaceSwap": "FAKE",
	# "NeuralTextures": "FAKE",
}

# Tham số cấu hình MVP
K_FRAMES = 10              # Lấy 10 frames/video
CROP_MARGIN = 0.30         # Mở rộng lề 30% xung quanh khuôn mặt
TARGET_SIZE = (224, 224)   # Size ảnh đầu ra 224x224
SEED = 42                  # Trọng số cố định cho split
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# Quality filter (để ảnh đủ sạch cho Phase 2)
ENABLE_QUALITY_FILTER = True

# MediaPipe face confidence (detection score)
FACE_CONF_THRESHOLD = 0.5

# Blur score = Laplacian variance (tính trên ảnh crop trước khi resize)
BLUR_THRESHOLD = 50.0

# Bbox area ratio = area(bbox) / area(image)
MIN_BBOX_AREA_RATIO = 0.001

# Khi đã có file ảnh .jpg ở disk:
# Nếu QUALITY filter bật: KHÔNG skip để đảm bảo metadata chất lượng thống nhất
# Nếu QUALITY filter tắt: skip để tăng tốc
SKIP_EXISTING_IF_QUALITY_DISABLED = True

# Khởi tạo Face Detection bằng MediaPipe Tasks
face_detector = None  # vision.FaceDetector


def init_face_detector():
	"""Lazy init của MediaPipe Tasks Face Detector."""
	global face_detector
	if face_detector is not None:
		return

	cache_dir = OUTPUT_BASE_DIR / "model_cache"
	cache_dir.mkdir(parents=True, exist_ok=True)
	model_path = cache_dir / "blaze_face_short_range.tflite"

	# Tự động tải model file chính thức của MediaPipe Tasks nếu chưa có
	if not model_path.exists():
		url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
		print(f"[MediaPipe Tasks] Downloading model: {url}")
		try:
			urllib.request.urlretrieve(url, str(model_path))
		except Exception as e:
			raise RuntimeError(f"[X] Không thể tải model MediaPipe Tasks. Lỗi: {e}")

	"""
	Đặt min_detection_confidence thấp (0.1) để MediaPipe không tự lọc bỏ sớm,
	-> Lấy được điểm face_conf thực tế để chủ động filter ở process_videos
	"""
	base_options = mp_python.BaseOptions(
			model_asset_path=str(model_path),
			# delegate=mp_python.BaseOptions.Delegate.GPU  # <-- DIESE ZEILE HINZUFÜGEN!
		)
	options = vision.FaceDetectorOptions(
		base_options=base_options,
		min_detection_confidence=0.1
	)
	face_detector = vision.FaceDetector.create_from_options(options)


# METADATA
def collect_video_files():
	video_records = []
	
	for folder_name, binary_label in DATASET_CONFIG.items():
		folder_path = BASE_ARCHIVE_DIR / folder_name
		if not folder_path.exists():
			print(f"[X] Không tìm thấy thư mục: {folder_path}")
			continue

		# Tạm bỏ qua các file ngoài "original" và "DeepFakeDetection"
		if folder_name != "original" and folder_name != "DeepFakeDetection":
			continue
			
		for video_path in folder_path.rglob("*.mp4"):
			rel_path = video_path.relative_to(folder_path)
			# Video_id cần unique theo toàn bộ đường dẫn tương đối (tránh collision khi có subfolder)
			rel_no_ext = rel_path.with_suffix("")
			video_id = f"{folder_name}/{rel_no_ext.as_posix()}"
			video_records.append({
				'video_id': video_id,          # Video identifier (unique)
				'method': folder_name,         # Nhãn phân loại cụ thể (original, DeepFakeDetection, ...)
				'label_binary': binary_label,  # Nhãn nhị phân (REAL, FAKE)
				'source_folder': folder_name,  # Folder gốc
				'full_path': str(video_path)   # Path đầy đủ tới file video
			})
			
	return pd.DataFrame(video_records)

# SPLIT DATASET
def split_dataset(df):
	train_df, temp_df = train_test_split(
		df, test_size=(VAL_RATIO + TEST_RATIO), stratify=df['label_binary'], random_state=SEED
	)
	
	val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
	val_df, test_df = train_test_split(
		temp_df, test_size=(1 - val_ratio_adjusted), stratify=temp_df['label_binary'], random_state=SEED
	)
	
	train_df = train_df.copy()
	val_df = val_df.copy()
	test_df = test_df.copy()
	
	train_df['split'] = 'train'
	val_df['split'] = 'val'
	test_df['split'] = 'test'
	
	return pd.concat([train_df, val_df, test_df], ignore_index=True)


def sanity_check_video_split(df_split: pd.DataFrame):
	# Chống leakage cùng video_id không được nằm ở nhiều split
	train_ids = set(df_split.loc[df_split['split'] == 'train', 'video_id'].tolist())
	val_ids = set(df_split.loc[df_split['split'] == 'val', 'video_id'].tolist())
	test_ids = set(df_split.loc[df_split['split'] == 'test', 'video_id'].tolist())

	if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
		dupe = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
		raise ValueError(f"[X] Leakage phát hiện: video_id xuất hiện nhiều split. Ví dụ: {list(dupe)[:10]}")


# XỬ LÝ CẮT MẶT & TRÍCH FRAME
def process_crop_face(frame_bgr, margin=0.30):
	h, w, _ = frame_bgr.shape
	global face_detector
	if face_detector is None:
		init_face_detector()

	# Chuyển đổi BGR sang RGB và bọc trong mp.Image
	rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
	mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

	detection_result = face_detector.detect(mp_image)
	if not detection_result.detections:
		return None

	# Chọn khuôn mặt có diện tích BBox lớn nhất
	best_detection = max(
		detection_result.detections,
		key=lambda d: d.bounding_box.width * d.bounding_box.height
	)

	bbox = best_detection.bounding_box
	bx, by, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

	# Lấy confidence score thực tế từ MediaPipe Tasks
	face_conf = float(best_detection.categories[0].score)
	bbox_area_ratio = float((bw * bh) / (w * h)) if (w * h) > 0 else 0.0

	# Expand margin
	mw = int(bw * margin)
	mh = int(bh * margin)
	x1 = max(0, bx - mw)
	y1 = max(0, by - mh)
	x2 = min(w, bx + bw + mw)
	y2 = min(h, by + bh + mh)

	face_crop = frame_bgr[y1:y2, x1:x2]
	if face_crop.size == 0:
		return None

	gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
	blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
	face_resized = cv2.resize(face_crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
	return face_resized, face_conf, float(blur_score), float(bbox_area_ratio)

def process_videos(df):
	extracted_records = []
	reject_stats = {}

	def reject(reason: str):
		reject_stats[reason] = reject_stats.get(reason, 0) + 1
	
	for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Video"):
		video_path = row['full_path']
		video_id = row['video_id']
		method = row['method']
		label_binary = row['label_binary']
		split = row['split']
		source_folder = row['source_folder']
		
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			print(f"\n [X] Không mở được file: {video_path}")
			continue
			
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		if total_frames <= 0:
			cap.release()
			continue

		# Lấy mẫu đều K frames
		if total_frames < K_FRAMES:
			frame_indices = list(range(total_frames))
		else:
			frame_indices = np.linspace(0, total_frames - 1, K_FRAMES, dtype=int)
			
		out_dir = OUTPUT_BASE_DIR / source_folder / split
		out_dir.mkdir(parents=True, exist_ok=True)
		
		safe_video_id = video_id.replace('/', '_').replace('\\', '_')
		
		for f_idx in frame_indices:
			img_name = f"{safe_video_id}_f{f_idx:04d}.jpg"
			img_save_path = out_dir / img_name

			# Nếu file ảnh đã tồn tại, chỉ skip khi QUALITY filter đang tắt
			if img_save_path.exists() and SKIP_EXISTING_IF_QUALITY_DISABLED and (not ENABLE_QUALITY_FILTER):
				extracted_records.append({
					'image_path': str(img_save_path),   # Path ảnh
					'label': method,                    # Deepfake method (Phase 2 đọc từ cột label)
					'label_binary': label_binary,       # REAL/FAKE
					'split': split,                     # Split: train/val/test
					'video_id': video_id,               # Video identifier
					'frame_index': f_idx,               # Frame index
					'face_conf': np.nan,                # Face confidence (MediaPipe score)
					'blur_score': np.nan,               # Blur score (Laplacian variance)
					'bbox_area_ratio': np.nan,          # BBox area ratio (bbox area / image area)
				})
				continue
				
			# Nếu chưa có ảnh mới đọc frame và gọi MediaPipe cắt mặt
			cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
			ret, frame = cap.read()
			if not ret or frame is None:
				reject('frame_read_failed')
				continue
				
			out = process_crop_face(frame, margin=CROP_MARGIN)
			if out is None:
				reject('no_face_detected')
				continue

			face_img, face_conf, blur_score, bbox_area_ratio = out
			
			if ENABLE_QUALITY_FILTER:
				if face_conf < FACE_CONF_THRESHOLD:
					reject('low_face_conf')
					continue
				if blur_score < BLUR_THRESHOLD:
					reject('low_blur')
					continue
				if bbox_area_ratio < MIN_BBOX_AREA_RATIO:
					reject('small_bbox')
					continue

			ok = cv2.imwrite(str(img_save_path), face_img)
			if not ok:
				reject('imwrite_failed')
				continue
			
			extracted_records.append({
				'image_path': str(img_save_path),  
				'label': method,
				'label_binary': label_binary,
				'split': split,
				'video_id': video_id,
				'frame_index': f_idx,
				'face_conf': face_conf,
				'blur_score': blur_score,
				'bbox_area_ratio': bbox_area_ratio,
			})
				
		cap.release()
		
	out_df = pd.DataFrame(extracted_records)

	# Xuất meta riêng cho phase 2
	meta_train_path = OUTPUT_BASE_DIR / "train_meta.csv"
	meta_val_path = OUTPUT_BASE_DIR / "val_meta.csv"
	meta_test_path = OUTPUT_BASE_DIR / "test_meta.csv"

	if len(out_df) == 0:
		print("\n[X] Không trích xuất được frame nào (out_df rỗng). Kiểm tra path dữ liệu và ngưỡng chất lượng.")
	else:
		for split_name, out_path in [
			('train', meta_train_path),
			('val', meta_val_path),
			('test', meta_test_path),
		]:
			split_df = out_df.loc[out_df['split'] == split_name].copy()
			split_df.to_csv(out_path, index=False)

	# Reject stats
	if len(reject_stats) > 0:
		reject_df = pd.DataFrame(
			{'reason': list(reject_stats.keys()), 'count': list(reject_stats.values())}
		).sort_values('count', ascending=False)
		reject_path = OUTPUT_BASE_DIR / 'reject_stats.csv'
		reject_df.to_csv(reject_path, index=False)
		print(f"\nReject stats lưu tại: {reject_path}")

	print("\n---> HOÀN TẤT!")
	print(f"Meta: {meta_train_path}, {meta_val_path}, {meta_test_path}")

	# Ghi chú cho Phase 2
	note_path = OUTPUT_BASE_DIR / 'PHASE2_NOTES.txt'
	note_lines = [
		"Phase 2 Notes (generated by process_data.py)",
		"",
		"1) Schema meta CSV:",
		"   - image_path: đường dẫn ảnh .jpg",
		"   - label: deepfake method (giá trị chính là tên thư mục: 'original' hoặc các tên deepfake khác)",
		"   - label_binary: 'REAL'/'FAKE' (tổng hợp để debug/monitor)",
		"   - split: train/val/test",
		"   - video_id, frame_index",
		"   - face_conf, blur_score, bbox_area_ratio (dùng cho sanity/điều chỉnh filter)",
		"",
		"2) Cách suy ra real/fake cho Phase 2:",
		"   - Phase 2 hiện chỉ đọc cột `label` (method deepfake) để suy ra REAL/FAKE.",
		"   - Nếu label == 'original' => REAL",
		"   - Nếu label == 'DeepFakeDetection' => FAKE",
		"   Lưu ý (MVP): ở bản hiện tại chỉ dùng mapping 'original' (REAL) và 'DeepFakeDetection' (FAKE).",
	]
	with open(note_path, 'w', encoding='utf-8') as f:
		f.write("\n".join(note_lines))
	print(f"Ghi chú cho Phase 2: {note_path}")


# EXECUTING PIPELINE
if __name__ == "__main__":
	print("1. Đang quét dữ liệu video...")
	df_videos = collect_video_files()
	print(f"   Tìm thấy: {len(df_videos)} video.")
	
	print("\n2. Phân chia tập dữ liệu (70/15/15) chống rò rỉ...")
	df_split = split_dataset(df_videos)
	sanity_check_video_split(df_split)
	
	print("\n--- Phân bố nhãn nhị phân theo Split ---")
	print(pd.crosstab(df_split['split'], df_split['label_binary']))
	
	print("\n--- Phân bố phương pháp theo Split ---")
	print(pd.crosstab(df_split['split'], df_split['method']))
	
	print("\n3. Trích xuất frame và cắt mặt...")
	process_videos(df_split)