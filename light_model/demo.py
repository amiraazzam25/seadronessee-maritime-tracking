"""
Light Model Demo — Unified Pipeline (YOLOv10n + MobileViT + GRU-SORT)
=====================================================================
Supports both Baseline (detect every frame) and Sparse (detect every N frames) modes.
By default, runs in Sparse mode (detect every 3rd frame) to minimize compute.

Usage:
    python demo.py <path_to_archive_or_folder> [output.mp4] [--skip-n N]

Example (Sparse - Default):
    python demo.py "data/DJI_0063_images.tar.gz"

Example (Baseline - Detect Every Frame):
    python demo.py "data/DJI_0063_images.tar.gz" --skip-n 1
"""

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DETECTOR_PATH   = "detector/Yolov10n_best.pt"
CLASSIFIER_PATH = "classifier/best_MobileVit.pth"
TELEMETRY_JSON  = None

DETECTION_CONF  = 0.30
USE_SAHI        = True
SAHI_SLICE_W    = 640
SAHI_SLICE_H    = 640
SAHI_OVERLAP    = 0.20
DEVICE          = "cuda"

USE_CLASSIFIER    = True   # Verified and trained
CLASSIFIER_THRESH = 0.70
NUM_CLASSES       = 5

TRACKER_MAX_AGE    = 60
TRACKER_MIN_HITS   = 3
TRACKER_IOU_THRESH = 0.30

SHOW_RAW_DETECTIONS = True
SHOW_PREDICTIONS    = True   # GRU predicted box (blue)
SHOW_MOTION_TRAIL   = True
TRAIL_LENGTH        = 10
OUTPUT_FPS          = 30
MAX_FRAMES          = None

# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
import cv2, json, zipfile, tarfile, sys, re, argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import timm
import torchvision.transforms as T
from ultralytics import YOLO as UltralyticsYOLO

from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel

sys.path.insert(0, str(Path(__file__).parent / "tracker"))
from sort_tracker import SortTracker

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_TELEMETRY = {k: 0.0 for k in
    ('gps_latitude', 'gps_longitude', 'altitude', 'gimbal_pitch',
     'compass_heading', 'xspeed', 'yspeed', 'zspeed')}

def frame_sort_key(p: Path):
    nums = re.findall(r'\d+', p.stem)
    return int(nums[-1]) if nums else 0

def extract_archive(archive_path: Path, extract_dir: Path):
    if extract_dir.exists():
        print(f"  [archive] Already extracted -> {extract_dir.name}"); return
    print(f"  [archive] Extracting {archive_path.name} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    suffix = ''.join(archive_path.suffixes).lower()
    if '.zip' in suffix:
        with zipfile.ZipFile(archive_path, 'r') as zf: zf.extractall(extract_dir)
    elif '.tar' in suffix or '.tgz' in suffix:
        mode = 'r:gz' if suffix.endswith('.gz') or suffix.endswith('.tgz') else 'r:'
        with tarfile.open(archive_path, mode) as tf: tf.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {suffix}")
    print("  [archive] Done.")

def load_telemetry(json_path, folder_name):
    try:
        data = json.load(open(json_path, 'r'))
        video_id = None
        for img in data['images']:
            if img['source'].get('folder_name') == folder_name:
                video_id = img['video_id']; break
        if video_id is None:
            print(f"  [telemetry] No entry for '{folder_name}' — zero telemetry."); return {}
        tmap = {}
        for img in data['images']:
            if img['video_id'] == video_id:
                m = img.get('meta', {})
                tmap[img['frame_index']] = {
                    'gps_latitude':    m.get('gps_latitude',    0.0),
                    'gps_longitude':   m.get('gps_longitude',   0.0),
                    'altitude':        m.get('altitude',         0.0),
                    'gimbal_pitch':    m.get('gimbal_pitch',     0.0),
                    'compass_heading': m.get('compass_heading',  0.0),
                    'xspeed':          m.get('xspeed',           0.0),
                    'yspeed':          m.get('yspeed',           0.0),
                    'zspeed':          m.get('zspeed',           0.0),
                }
        print(f"  [telemetry] Loaded {len(tmap)} frames (video_id={video_id}).")
        return tmap
    except Exception as e:
        print(f"  [telemetry] Failed ({e}) — zero telemetry."); return {}

def render_frame(frame, active_tracks, yolo_boxes, tracker, colors,
                 is_coast, show_raw, show_pred, show_trail, trail_len):
    if is_coast:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 140, 255), 6)

    if show_trail:
        for tk in tracker.tracks:
            if tk.time_since_update == 0 and len(tk.history) > 1 and tk.id in colors:
                col = colors[tk.id]
                hist = tk.history[-trail_len:]
                for i in range(1, len(hist)):
                    p, c = hist[i-1], hist[i]
                    p1 = (int(p[0]+p[2]/2), int(p[1]+p[3]/2))
                    p2 = (int(c[0]+c[2]/2), int(c[1]+c[3]/2))
                    cv2.line(frame, p1, p2, col, 2)

    if show_raw and not is_coast:
        for bx, by, bw, bh, _, score in yolo_boxes:
            cv2.rectangle(frame, (int(bx), int(by)), (int(bx+bw), int(by+bh)), (0, 0, 220), 1)
            cv2.putText(frame, f"{score:.2f}", (int(bx), int(by+bh)+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)

    if show_pred:
        for tk in tracker.tracks:
            if hasattr(tk, 'predicted_bbox') and tk.predicted_bbox is not None:
                px, py, pw, ph = [int(v) for v in tk.predicted_bbox]
                cv2.rectangle(frame, (px, py), (px+pw, py+ph), (220, 80, 0), 1)

    for x, y, w, h, tid, cls_name in active_tracks:
        if tid not in colors:
            np.random.seed(tid)
            colors[tid] = tuple(int(c) for c in np.random.randint(80, 240, 3))
        col = colors[tid]
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), col, 3)
        label = cls_name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        by_top = max(y - 30, 0)
        cv2.rectangle(frame, (x, by_top), (x+tw+10, by_top+28), col, -1)
        cv2.putText(frame, label, (x+5, by_top+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    mode_str = "COAST (GRU)" if is_coast else "DETECT"
    mode_col = (0, 140, 255) if is_coast else (100, 220, 100)
    cv2.rectangle(frame, (8, 8), (310, 78), (20, 20, 20), -1)
    cv2.putText(frame, f"Mode       : {mode_str}", (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_col, 2)
    cv2.putText(frame, f"Tracks     : {len(active_tracks)}", (14, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 255), 2)
    return frame

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run(input_path: str, output_path: str = None, skip_n: int = 3):
    HERE     = Path(__file__).parent
    src_path = Path(input_path)

    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}
    if src_path.is_dir():
        images = sorted([p for p in src_path.rglob('*') if p.suffix.lower() in IMAGE_EXTS],
                        key=frame_sort_key)
        stem = src_path.name
    else:
        stem = src_path.name.replace(''.join(src_path.suffixes), '')
        extract_dir = src_path.parent / stem
        extract_archive(src_path, extract_dir)
        images = sorted([p for p in extract_dir.rglob('*') if p.suffix.lower() in IMAGE_EXTS],
                        key=frame_sort_key)

    if not images:
        print("ERROR: No images found."); return
    print(f"  [pipeline] Found {len(images)} frames.")

    telemetry_map = {}
    if TELEMETRY_JSON and Path(TELEMETRY_JSON).exists():
        telemetry_map = load_telemetry(TELEMETRY_JSON, stem.replace('_images', ''))

    print("\n  [models] Loading detector (YOLOv10n) ...")
    if USE_SAHI:
        detector = UltralyticsDetectionModel(
            model_path=str(HERE / DETECTOR_PATH),
            confidence_threshold=DETECTION_CONF,
            device=DEVICE,
        )
    else:
        detector = UltralyticsYOLO(str(HERE / DETECTOR_PATH))

    classifier = None; vit_transform = None; idx_to_class = None
    if USE_CLASSIFIER:
        print("  [models] Loading classifier (MobileViT) ...")
        try:
            classifier = timm.create_model("mobilevit_s", pretrained=False, num_classes=NUM_CLASSES)
            ckpt = torch.load(str(HERE / CLASSIFIER_PATH), map_location=DEVICE)
            classifier.load_state_dict(ckpt['model_state_dict'])
            classifier.to(DEVICE).eval()
            idx_to_class = ckpt['idx_to_class']
            vit_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("  [models] Loaded MobileViT classifier successfully.")
        except Exception as e:
            print(f"  [models] WARNING: Failed to load MobileViT: {e}")
            classifier = None

    print("  [models] Loading GRU-SORT tracker ...")
    tracker = SortTracker(max_age=TRACKER_MAX_AGE, min_hits=TRACKER_MIN_HITS,
                          iou_threshold=TRACKER_IOU_THRESH)

    sample  = cv2.imread(str(images[0]))
    h_vid, w_vid = sample.shape[:2]
    
    mode_lbl = "baseline" if skip_n == 1 else "sparse"
    if output_path is None:
        out_dir = src_path.parent / "outputs"; out_dir.mkdir(exist_ok=True)
        v = 1
        while (out_dir / f"light_{mode_lbl}_{stem}_v{v}.mp4").exists(): v += 1
        output_path = str(out_dir / f"light_{mode_lbl}_{stem}_v{v}.mp4")
        
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             OUTPUT_FPS, (w_vid, h_vid))

    print(f"\n  [output] {output_path}  ({w_vid}x{h_vid} @ {OUTPUT_FPS} fps)")
    print(f"  [config] SAHI={USE_SAHI}  skip_N={skip_n}  "
          f"max_age={TRACKER_MAX_AGE}\n")

    colors = {}
    n_frames = len(images) if MAX_FRAMES is None else min(MAX_FRAMES, len(images))
    detect_count = coast_count = 0

    for frame_idx, img_path in enumerate(images[:n_frames]):
        frame = cv2.imread(str(img_path))
        if frame is None: continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb)
        
        is_coast  = (frame_idx % skip_n != 0)
        telemetry = telemetry_map.get(frame_idx, DEFAULT_TELEMETRY)

        if not is_coast:
            detect_count += 1
            if USE_SAHI:
                result = get_sliced_prediction(
                    pil_img, detector,
                    slice_height=SAHI_SLICE_H, slice_width=SAHI_SLICE_W,
                    overlap_height_ratio=SAHI_OVERLAP, overlap_width_ratio=SAHI_OVERLAP,
                    verbose=0,
                )
                raw_boxes = [(pred.bbox.minx, pred.bbox.miny,
                              pred.bbox.maxx-pred.bbox.minx, pred.bbox.maxy-pred.bbox.miny,
                              pred.category.name, pred.score.value)
                             for pred in result.object_prediction_list]
            else:
                res = detector(pil_img, conf=DETECTION_CONF, device=DEVICE, verbose=False)[0]
                raw_boxes = [(b.xyxy[0][0], b.xyxy[0][1],
                              b.xyxy[0][2]-b.xyxy[0][0], b.xyxy[0][3]-b.xyxy[0][1],
                              res.names[int(b.cls)], float(b.conf))
                             for b in res.boxes]

            yolo_boxes = []
            for bx, by, bw, bh, cls_name, score in raw_boxes:
                if USE_CLASSIFIER and classifier is not None and score < CLASSIFIER_THRESH:
                    x1, y1 = max(0, int(bx)), max(0, int(by))
                    x2, y2 = min(w_vid, int(bx+bw)), min(h_vid, int(by+bh))
                    if x2 > x1 and y2 > y1:
                        crop = pil_img.crop((x1, y1, x2, y2))
                        with torch.no_grad():
                            out = classifier(vit_transform(crop).unsqueeze(0).to(DEVICE))
                            vit_cls = idx_to_class[out.argmax(1).item()]
                        if vit_cls != cls_name:
                            cls_name = f"{vit_cls} (cls)"
                yolo_boxes.append([bx, by, bw, bh, cls_name, score])
            active_tracks = tracker.update(yolo_boxes, telemetry, frame_size=(w_vid, h_vid))
        else:
            coast_count  += 1
            yolo_boxes    = []
            active_tracks = tracker.update_coast(telemetry)

        frame = render_frame(frame, active_tracks, yolo_boxes, tracker, colors,
                             is_coast, SHOW_RAW_DETECTIONS, SHOW_PREDICTIONS,
                             SHOW_MOTION_TRAIL, TRAIL_LENGTH)

        h, w = frame.shape[:2]
        cv2.putText(frame, f"Frame {frame_idx+1}/{n_frames}", (w-250, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        writer.write(frame)
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx+1}/{n_frames}  |  "
                  f"{'DETECT' if not is_coast else 'COAST ':6s}  "
                  f"trk={len(active_tracks)}")

    writer.release()
    total = detect_count + coast_count
    print(f"\n  [done] Saved -> {output_path}")
    print(f"  [stats] Detection : {detect_count}/{total} ({100*detect_count/total:.1f}%)")
    if total > 0 and coast_count > 0:
        print(f"          Coast     : {coast_count}/{total}  ({100*coast_count/total:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    DEFAULT_INPUT = r"C:\Users\youse\Downloads\Helwan\3rd\Neural Network\Seedrone\data\DJI_0063_images.tar.gz"
    
    parser = argparse.ArgumentParser(description="Light Model Tracker Demo (Baseline & Sparse modes).")
    parser.add_argument("input_path", nargs="?", default=DEFAULT_INPUT, 
                        help="Path to DJI frames archive (zip/tar.gz) or images directory.")
    parser.add_argument("output_path", nargs="?", default=None, 
                        help="Output path for generated annotated MP4.")
    parser.add_argument("--skip-n", type=int, default=3, 
                        help="Run object detector only every Nth frame. The GRU coasts in-between. "
                             "Set to 1 to run detector on EVERY frame (Baseline mode). Default is 3 (Sparse).")
    
    args = parser.parse_args()
    
    mode_lbl = "Sparse" if args.skip_n > 1 else "Baseline"
    print(f"\n{'='*60}")
    print(f"  Light Model Demo — {mode_lbl} Mode")
    print(f"  Input : {args.input_path}")
    if args.skip_n > 1:
        print(f"  Skip  : Run detection every {args.skip_n} frames.")
    else:
        print(f"  Skip  : None (Detector active every frame).")
    print(f"{'='*60}\n")
    
    run(args.input_path, args.output_path, skip_n=args.skip_n)
