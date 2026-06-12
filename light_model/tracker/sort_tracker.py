import os
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import joblib

# ---------------------------------------------------------
# 1. Hungarian Matching Metric (Centroid Distance + IoU)
# ---------------------------------------------------------
def bbox_iou(box1, box2):
    x1_min, y1_min, w1, h1 = box1[:4]
    x2_min, y2_min, w2, h2 = box2[:4]
    
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    
    inter_area = inter_w * inter_h
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    return inter_area / union_area

def compute_cost(pred_box, det_box, frame_diagonal):
    cx1, cy1 = pred_box[0] + pred_box[2]/2, pred_box[1] + pred_box[3]/2
    cx2, cy2 = det_box[0] + det_box[2]/2, det_box[1] + det_box[3]/2
    dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    norm_dist = dist / frame_diagonal
    
    iou = bbox_iou(pred_box, det_box)
    iou_cost = 1.0 - iou
    
    return norm_dist + iou_cost

# ---------------------------------------------------------
# 2. GRU Neural Network
# ---------------------------------------------------------
class GRUTracker(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# ---------------------------------------------------------
# 3. Track Object
# ---------------------------------------------------------
class Track:
    _id_counter = 1
    
    def __init__(self, initial_bbox, initial_telemetry, class_name="Unknown"):
        self.id = Track._id_counter
        Track._id_counter += 1
        
        self.history = [] 
        self.add_to_history(initial_bbox, initial_telemetry)
        
        self.time_since_update = 0
        self.hits = 1
        self.predicted_bbox = initial_bbox
        self.class_name = class_name
        
    def add_to_history(self, bbox, telemetry):
        feature_vector = [
            bbox[0], bbox[1], bbox[2], bbox[3],
            telemetry.get('gps_latitude', 0.0),
            telemetry.get('gps_longitude', 0.0),
            telemetry.get('altitude', 0.0),
            telemetry.get('gimbal_pitch', 0.0),
            telemetry.get('compass_heading', 0.0),
            telemetry.get('xspeed', 0.0),
            telemetry.get('yspeed', 0.0),
            telemetry.get('zspeed', 0.0)
        ]
        self.history.append(feature_vector)
        if len(self.history) > 10:
            self.history.pop(0)

# ---------------------------------------------------------
# 4. Light SORT Tracker Engine (GRU)
# ---------------------------------------------------------
class SortTracker:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        base_dir = Path(__file__).parent
        model_path = base_dir / 'best_gru.pth'
        scaler_X_path = base_dir / 'scaler_X.pkl'
        scaler_Y_path = base_dir / 'scaler_Y.pkl'
        
        # 1. Initialize PyTorch Model
        self.model = GRUTracker(input_size=12).to(self.device)
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"[Light Tracker] Loaded GRU weights from {model_path.name}")
        else:
            print(f"WARNING: Weights not found at {model_path}. Using random weights!")
            
        # 2. Load the Scalers
        if scaler_X_path.exists() and scaler_Y_path.exists():
            self.scaler_X = joblib.load(scaler_X_path)
            self.scaler_Y = joblib.load(scaler_Y_path)
            print(f"[Light Tracker] Scalers loaded.")
        else:
            print(f"WARNING: Scalers not found at {scaler_X_path}!")
        
    def predict_track(self, track, use_nn=True):
        if not use_nn or len(track.history) < 10:
            track.predicted_bbox = track.history[-1][:4]
            return track.predicted_bbox
            
        hist = np.array(track.history) # shape (10, 12)
        vels = np.zeros((9, 12))
        vels[:, 0] = hist[1:, 0] - hist[:-1, 0] # dx
        vels[:, 1] = hist[1:, 1] - hist[:-1, 1] # dy
        vels[:, 2:] = hist[1:, 2:] # w, h, and 8 telemetry
        
        scaled_vels = self.scaler_X.transform(vels)
        X_tensor = torch.tensor(scaled_vels, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
            
        y_pred_vel = self.scaler_Y.inverse_transform(y_pred_scaled)[0]
        
        last_abs = hist[-1, :4]
        pred_bbox = [
            last_abs[0] + y_pred_vel[0],
            last_abs[1] + y_pred_vel[1],
            y_pred_vel[2],
            y_pred_vel[3]
        ]
        track.predicted_bbox = pred_bbox
        return pred_bbox
        
    def update(self, detections, telemetry, use_nn=True, frame_size=(3840, 2160)):
        frame_diagonal = np.sqrt(frame_size[0] ** 2 + frame_size[1] ** 2)
        predictions = []
        for track in self.tracks:
            pred = self.predict_track(track, use_nn=use_nn)
            predictions.append(pred)
            
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        if len(self.tracks) > 0 and len(detections) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for t, pred_box in enumerate(predictions):
                for d, det_box in enumerate(detections):
                    cost_matrix[t, d] = compute_cost(pred_box, det_box, frame_diagonal)
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for t, d in zip(row_ind, col_ind):
                if cost_matrix[t, d] > 0.25:
                    continue
                matched_indices.append((t, d))
                unmatched_tracks.remove(t)
                unmatched_detections.remove(d)
                
        for t, d in matched_indices:
            self.tracks[t].add_to_history(detections[d][:4], telemetry)
            self.tracks[t].class_name = detections[d][4]
            self.tracks[t].time_since_update = 0
            self.tracks[t].hits += 1
            
        for t in unmatched_tracks:
            self.tracks[t].time_since_update += 1
            
        for d in unmatched_detections:
            new_track = Track(detections[d][:4], telemetry, class_name=detections[d][4])
            self.tracks.append(new_track)
            
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        outputs = []
        for t in self.tracks:
            if t.hits >= self.min_hits or t.time_since_update == 0:
                current_box = t.history[-1][:4]
                outputs.append([current_box[0], current_box[1], current_box[2], current_box[3], t.id, t.class_name])
                
        return outputs

    def update_coast(self, telemetry, use_nn=True):
        outputs = []
        for track in self.tracks:
            pred_box = self.predict_track(track, use_nn=use_nn)
            track.add_to_history(pred_box, telemetry)
            
            if track.hits >= self.min_hits or track.time_since_update == 0:
                current_box = track.history[-1][:4]
                outputs.append([current_box[0], current_box[1], current_box[2], current_box[3], track.id, track.class_name])
                
        return outputs
