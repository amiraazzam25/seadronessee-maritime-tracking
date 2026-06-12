"""
Light Model Inference API
=========================
Provides a programmatic interface to run the light pipeline (YOLOv10n + MobileViT + GRU-SORT)
on individual frames without saving output videos.
"""
import cv2, sys
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

class LightInferencePipeline:
    def __init__(self, 
                 detector_path="detector/Yolov10n_best.pt", 
                 classifier_path="classifier/best_MobileVit.pth", 
                 use_sahi=True, 
                 use_classifier=True,
                 device=None):
        
        self.here = Path(__file__).parent
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_sahi = use_sahi
        self.use_classifier = use_classifier
        
        # Load Detector
        print("[Light API] Loading detector...")
        full_detector_path = str(self.here / detector_path)
        if use_sahi:
            self.detector = UltralyticsDetectionModel(
                model_path=full_detector_path,
                confidence_threshold=0.3,
                device=self.device
            )
        else:
            self.detector = UltralyticsYOLO(full_detector_path)
            
        # Load Classifier
        self.classifier = None
        if use_classifier:
            print("[Light API] Loading MobileViT classifier...")
            try:
                self.classifier = timm.create_model("mobilevit_s", pretrained=False, num_classes=5)
                ckpt = torch.load(str(self.here / classifier_path), map_location=self.device)
                self.classifier.load_state_dict(ckpt['model_state_dict'])
                self.classifier.to(self.device).eval()
                self.idx_to_class = ckpt['idx_to_class']
                self.vit_transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                print("[Light API] Loaded MobileViT weights successfully.")
            except Exception as e:
                print(f"[Light API] WARNING: Failed to load MobileViT: {e}")
                self.use_classifier = False
            
        # Load Tracker
        print("[Light API] Initializing tracker...")
        self.tracker = SortTracker(max_age=60, min_hits=3, iou_threshold=0.30)
        
    def process_frame(self, frame_bgr, telemetry=None, is_coast_frame=False):
        """
        Processes a single BGR image.
        Returns: List of active tracks formatted as [[x, y, w, h, track_id, class_name], ...]
        """
        h_vid, w_vid = frame_bgr.shape[:2]
        telemetry = telemetry or {k: 0.0 for k in ('gps_latitude', 'gps_longitude', 'altitude', 'gimbal_pitch',
                                                   'compass_heading', 'xspeed', 'yspeed', 'zspeed')}
        
        if is_coast_frame:
            return self.tracker.update_coast(telemetry)
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # 1. Detection
        if self.use_sahi:
            result = get_sliced_prediction(
                pil_img, self.detector,
                slice_height=640, slice_width=640,
                overlap_height_ratio=0.20, overlap_width_ratio=0.20,
                verbose=0
            )
            raw_boxes = []
            for pred in result.object_prediction_list:
                bb = pred.bbox
                raw_boxes.append((bb.minx, bb.miny, bb.maxx - bb.minx,
                                  bb.maxy - bb.miny, pred.category.name, pred.score.value))
        else:
            res = self.detector(pil_img, conf=0.30, device=self.device, verbose=False)[0]
            raw_boxes = []
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw_boxes.append((x1, y1, x2-x1, y2-y1,
                                  res.names[int(box.cls)], float(box.conf)))
                                  
        # 2. Classifier Verification
        yolo_boxes = []
        for bx, by, bw, bh, cls_name, score in raw_boxes:
            if self.use_classifier and self.classifier and score < 0.70:
                x1, y1 = max(0, int(bx)), max(0, int(by))
                x2, y2 = min(w_vid, int(bx+bw)), min(h_vid, int(by+bh))
                if x2 > x1 and y2 > y1:
                    crop = pil_img.crop((x1, y1, x2, y2))
                    with torch.no_grad():
                        out = self.classifier(self.vit_transform(crop).unsqueeze(0).to(self.device))
                        vit_cls = self.idx_to_class[out.argmax(1).item()]
                    if vit_cls != cls_name:
                        cls_name = f"{vit_cls} (cls)"
            yolo_boxes.append([bx, by, bw, bh, cls_name, score])
            
        # 3. Tracking Update
        active_tracks = self.tracker.update(yolo_boxes, telemetry, frame_size=(w_vid, h_vid))
        return active_tracks
