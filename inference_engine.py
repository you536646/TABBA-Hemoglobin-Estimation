"""
=============================================================================
TABBA Web Inference Engine
=============================================================================
Ported from tabba_v61_enhanced_extractor.py for real-time web inference.

Pipeline per frame:
  1. YOLO detection → NMS dedup → distance filter (nearest 8 patches)
  2. Angular sort (clockwise from 12 o'clock) + White-anchor rotation (P1=WHITE)
  3. Safe Core extraction (inner 30% crop + P5-P95 outlier rejection)
  4. 142-feature engineering (RGB, HSV, LAB, OD, Deltas, Ratios, Advanced)
  5. XGBoost prediction

Session Protocol (Chemistry-Correct):
  Phase 1 (0-10s): Predict per frame for UI feedback only (medically discarded)
  Phase 2 (10-15s): Accumulate 142 raw features per frame
  Final Verdict (t=15s): Mean-average 142 features → single XGBoost call
=============================================================================
"""

import os
import math
import cv2
import numpy as np
import joblib
import time

# Fix OpenMP DLL collision between PyTorch and XGBoost on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
from ultralytics import YOLO


# =============================================================================
# CONFIG
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PARENT_DIR, 'best.pt')
ML_MODEL_PATH = os.path.join(PARENT_DIR, 'tabba_brain_v66_kaggle.pkl')

# Detection
YOLO_CONF = 0.20
BLOOD_MIN_CONF = 0.25
NMS_IOU_THRESH = 0.30
NUM_PATCHES = 8

# Safe Core
CORE_SHRINK = 0.35
PERCENTILE_LOW = 5
PERCENTILE_HIGH = 95

# Session
SESSION_DURATION = 15.0      # Total seconds
STABILIZATION_TIME = 10.0    # Phase 1 ends at 10s
CAPTURE_FPS = 8              # Expected frames per second


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================
def compute_iou(box1, box2):
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)


def apply_nms(patches, iou_thresh=0.3):
    patches_sorted = sorted(patches, key=lambda x: x['conf'], reverse=True)
    keep = []
    for p in patches_sorted:
        is_duplicate = False
        for kept in keep:
            if compute_iou(p['xyxy'], kept['xyxy']) > iou_thresh:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(p)
    return keep


def get_center(xyxy):
    return ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)


def calc_distance(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


# =============================================================================
# SAFE CORE EXTRACTION (Scale-Invariant)
# =============================================================================
def extract_safe_core(img_rgb, x1, y1, x2, y2):
    """
    Extract color from inner 30% of bounding box.
    - Crop inner 30%: remove 35% from each side
    - P5-P95 outlier rejection on brightness
    - Median extraction per channel (RGB, HSV, LAB)
    """
    w, h = x2 - x1, y2 - y1
    nx1 = int(x1 + w * CORE_SHRINK)
    ny1 = int(y1 + h * CORE_SHRINK)
    nx2 = int(x1 + w * (1.0 - CORE_SHRINK))
    ny2 = int(y1 + h * (1.0 - CORE_SHRINK))

    if nx2 <= nx1 or ny2 <= ny1:
        return None

    roi_rgb = img_rgb[ny1:ny2, nx1:nx2]
    if roi_rgb.size == 0 or roi_rgb.shape[0] < 2 or roi_rgb.shape[1] < 2:
        return None

    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    roi_lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)

    brightness = np.mean(roi_rgb, axis=2)
    p_low = np.percentile(brightness, PERCENTILE_LOW)
    p_high = np.percentile(brightness, PERCENTILE_HIGH)
    mask = (brightness >= p_low) & (brightness <= p_high)

    total_px = roi_rgb.shape[0] * roi_rgb.shape[1]
    valid_count = int(np.sum(mask))
    if valid_count < 5:
        mask = np.ones(brightness.shape, dtype=bool)
        valid_count = total_px

    purity = round(valid_count / max(total_px, 1) * 100, 1)

    valid_rgb = roi_rgb[mask]
    valid_hsv = roi_hsv[mask]
    valid_lab = roi_lab[mask]

    med_rgb = np.median(valid_rgb, axis=0).astype(int)
    med_hsv = np.median(valid_hsv, axis=0).astype(int)
    med_lab = np.median(valid_lab, axis=0).astype(int)

    return {
        'R': int(med_rgb[0]), 'G': int(med_rgb[1]), 'B': int(med_rgb[2]),
        'H': int(med_hsv[0]), 'S': int(med_hsv[1]), 'V': int(med_hsv[2]),
        'L': int(med_lab[0]), 'a': int(med_lab[1]), 'b': int(med_lab[2]),
        'purity': purity,
        'core_coords': (nx1, ny1, nx2, ny2),
    }


# =============================================================================
# ANGULAR SORTING + WHITE ANCHOR
# =============================================================================
def angular_sort_with_anchor(blood_center, patches, img_rgb):
    """
    Sort patches clockwise from 12 o'clock, rotate so WHITE is P1.
    """
    cx, cy = blood_center

    for p in patches:
        px, py = get_center(p['xyxy'])
        dx, dy = px - cx, py - cy
        angle = math.atan2(dx, -dy)
        if angle < 0:
            angle += 2 * math.pi
        p['angle'] = angle

    patches.sort(key=lambda x: x['angle'])

    best_idx = 0
    best_score = -999

    for i, p in enumerate(patches):
        x1, y1, x2, y2 = p['xyxy']
        w, h = x2 - x1, y2 - y1
        cx1 = int(x1 + w * 0.3)
        cy1 = int(y1 + h * 0.3)
        cx2 = int(x2 - w * 0.3)
        cy2 = int(y2 - h * 0.3)

        if cx2 <= cx1 or cy2 <= cy1:
            continue

        roi = img_rgb[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mean_v = float(np.mean(hsv[:, :, 2]))
        mean_s = float(np.mean(hsv[:, :, 1]))
        white_score = mean_v - mean_s * 0.5

        if white_score > best_score:
            best_score = white_score
            best_idx = i

    rotated = patches[best_idx:] + patches[:best_idx]

    return rotated


# =============================================================================
# 142-FEATURE ENGINEERING (from v61_enhanced)
# =============================================================================
def build_feature_names():
    """Build the ordered list of 142 feature column names (excluding metadata).
    NOTE: Blood_Area and Blood_Purity are NOT model features — they are metadata.
    The 142 features = 9 blood colors + 72 patch colors + 3 OD + 4 advanced OD +
    3 RGB stats + 2 LAB advanced + 1 LS + 24 deltas + 24 ratios.
    """
    cols = []

    ch_names = ['R', 'G', 'B', 'H', 'S', 'V', 'L', 'a', 'b']

    # Blood colors (9)
    for ch in ch_names:
        cols.append(f'Blood_{ch}')

    # Patch colors: P1-P8 x 9 channels (72)
    for i in range(1, 9):
        for ch in ch_names:
            cols.append(f'P{i}_{ch}')

    # OD Basic (3)
    cols.extend(['OD_R', 'OD_G', 'OD_B'])

    # OD Advanced (4)
    cols.extend(['OD_Total', 'OD_Red_Ratio', 'OD_RG_Ratio', 'OD_Variance'])

    # RGB Stats (3)
    cols.extend(['Blood_Red_Dom', 'Blood_RG_Diff', 'Blood_RGB_Std'])

    # LAB Advanced (2)
    cols.extend(['Blood_Chroma', 'Blood_Hue_Deg'])

    # LS Interaction (1)
    cols.append('Blood_LS_Product')

    # LAB Deltas: Blood - Pi for L, a, b (24)
    for i in range(1, 9):
        for ch in ['L', 'a', 'b']:
            cols.append(f'Delta_{ch}_P{i}')

    # RGB Ratios: Blood / Pi for R, G, B (24)
    for i in range(1, 9):
        for ch in ['R', 'G', 'B']:
            cols.append(f'Ratio_{ch}_P{i}')

    return cols


def compute_features_enhanced(blood_colors, patch_colors_list):
    """
    Compute ALL 142 features including 10 advanced chemical features.
    Returns dict keyed by feature name.
    """
    features = {}
    p1 = patch_colors_list[0]  # P1 = WHITE (guaranteed by anchor)

    # === Basic OD (3) ===
    for ch in ['R', 'G', 'B']:
        blood_val = max(blood_colors[ch], 1)
        ref_val = max(p1[ch], 1)
        features[f'OD_{ch}'] = round(-math.log10(blood_val / ref_val), 4)

    # === Advanced OD (4) ===
    features['OD_Total'] = features['OD_R'] + features['OD_G'] + features['OD_B']
    features['OD_Red_Ratio'] = round(features['OD_R'] / max(features['OD_Total'], 1e-6), 4)
    features['OD_RG_Ratio'] = round(features['OD_R'] / max(features['OD_G'], 1e-6), 4)
    features['OD_Variance'] = round(float(np.var([features['OD_R'], features['OD_G'], features['OD_B']])), 4)

    # === RGB Stats (3) ===
    rgb_vals = [blood_colors['R'], blood_colors['G'], blood_colors['B']]
    features['Blood_Red_Dom'] = round(blood_colors['R'] / (sum(rgb_vals) + 1e-6), 4)
    features['Blood_RG_Diff'] = blood_colors['R'] - blood_colors['G']
    features['Blood_RGB_Std'] = round(float(np.std(rgb_vals)), 2)

    # === LAB Advanced (2) ===
    features['Blood_Chroma'] = round(math.sqrt(blood_colors['a']**2 + blood_colors['b']**2), 2)
    hue_rad = math.atan2(blood_colors['b'], blood_colors['a'])
    features['Blood_Hue_Deg'] = round((hue_rad * 180 / math.pi) % 360, 2)

    # === LS Interaction (1) ===
    features['Blood_LS_Product'] = blood_colors['L'] * blood_colors['S']

    # === LAB Deltas (24) ===
    for i, p in enumerate(patch_colors_list, 1):
        for ch in ['L', 'a', 'b']:
            features[f'Delta_{ch}_P{i}'] = blood_colors[ch] - p[ch]

    # === RGB Ratios (24) ===
    for i, p in enumerate(patch_colors_list, 1):
        for ch in ['R', 'G', 'B']:
            features[f'Ratio_{ch}_P{i}'] = round(blood_colors[ch] / max(p[ch], 1), 4)

    return features


def extract_all_features(blood_box, sorted_patches):
    """
    From detected blood_box and sorted_patches, extract the full 142-feature vector.
    Returns dict with all feature names as keys.
    """
    blood_c = blood_box['colors']
    patch_colors_list = [p['colors'] for p in sorted_patches]

    row = {
        'Blood_Area': blood_box['area'],
        'Blood_Purity': blood_c.get('purity', 100.0),
    }

    # Blood direct colors
    for ch in ['R', 'G', 'B', 'H', 'S', 'V', 'L', 'a', 'b']:
        row[f'Blood_{ch}'] = blood_c[ch]

    # Patch direct colors
    for i, pc in enumerate(patch_colors_list, 1):
        for ch in ['R', 'G', 'B', 'H', 'S', 'V', 'L', 'a', 'b']:
            row[f'P{i}_{ch}'] = pc[ch]

    # Engineered features (OD, Deltas, Ratios, Advanced)
    eng = compute_features_enhanced(blood_c, patch_colors_list)
    row.update(eng)

    return row


# =============================================================================
# YOLO DETECTOR
# =============================================================================
class YOLODetector:
    def __init__(self):
        print(f"[InferenceEngine] Loading YOLO model from {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        print(f"[InferenceEngine] YOLO classes: {self.model.names}")

    def detect(self, img_rgb):
        """
        Run YOLO detection on an RGB image.
        Returns (blood_box, sorted_patches, status).
        """
        results = self.model.predict(img_rgb, conf=YOLO_CONF, verbose=False)
        result = results[0]

        blood_box = None
        all_patches = []

        for box in result.boxes:
            label = self.model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            area = (x2 - x1) * (y2 - y1)

            if label == 'blood' and conf >= BLOOD_MIN_CONF:
                if blood_box is None or conf > blood_box['conf']:
                    blood_box = {'xyxy': (x1, y1, x2, y2), 'conf': conf, 'area': area}
            elif label == 'patch':
                all_patches.append({'xyxy': (x1, y1, x2, y2), 'conf': conf, 'area': area})

        if blood_box is None:
            return None, None, 'no_blood'

        # NMS
        all_patches = apply_nms(all_patches, NMS_IOU_THRESH)

        if len(all_patches) < NUM_PATCHES:
            return None, None, f'only_{len(all_patches)}_patches'

        # Distance filter: nearest 8
        blood_center = get_center(blood_box['xyxy'])
        for p in all_patches:
            p['dist'] = calc_distance(blood_center, get_center(p['xyxy']))
        all_patches.sort(key=lambda x: x['dist'])
        nearest = all_patches[:NUM_PATCHES]

        # Angular sort + White anchor
        sorted_patches = angular_sort_with_anchor(blood_center, nearest, img_rgb)

        # Extract colors
        bx1, by1, bx2, by2 = blood_box['xyxy']
        blood_colors = extract_safe_core(img_rgb, bx1, by1, bx2, by2)
        if blood_colors is None:
            return None, None, 'blood_core_empty'
        blood_box['colors'] = blood_colors

        for p in sorted_patches:
            px1, py1, px2, py2 = p['xyxy']
            p_colors = extract_safe_core(img_rgb, px1, py1, px2, py2)
            if p_colors is None:
                return None, None, 'patch_core_empty'
            p['colors'] = p_colors

        return blood_box, sorted_patches, 'ok'


# =============================================================================
class HemoglobinPredictor:
    def __init__(self):
        print("[InferenceEngine] Mocking Hemoglobin Predictor (Bypassing XGBoost CPU Crash)")
        self.feature_names = build_feature_names()
        print(f"[InferenceEngine] Using {len(self.feature_names)} features for dummy prediction")

    def predict(self, feature_dict):
        """
        Mock Hemoglobin prediction. Since XGBoost crashes on this laptop's CPU,
        this dummy logic calculates a proportional Hg value based on Blood OD and Deltas,
        just so the UI can be tested seamlessly.
        """
        od_total = feature_dict.get('OD_Total', 0)
        blood_red = feature_dict.get('Blood_R', 100)
        
        # Simple logical baseline calculation for demo purposes
        base_hg = 10.0 + (blood_red / 255.0) * 4.0 - (od_total * 0.5)
        
        # Add a tiny bit of random noise for realism
        base_hg += np.random.uniform(-0.2, 0.2)
        
        return float(np.clip(base_hg, 7.0, 18.0))


# =============================================================================
# SESSION MANAGER (Chemistry-Correct Protocol)
# =============================================================================
class SessionManager:
    """
    Manages the 5-second kinetic session.

    Phase 1 (0-0s): Skipped
    Phase 2 (0-5s): Accumulate 142 features per frame
    Final Verdict (t=5s): Mean features → single XGBoost call
    """

    def __init__(self):
        self.reset()
        self.feature_names = build_feature_names()

    def reset(self):
        self.start_time = None
        self.phase2_features = []  # List of feature dicts (Phase 2 accumulation)
        self.is_finalized = False
        self.final_result = None
        self.frame_count = 0
        self.phase2_frame_count = 0

    def start(self):
        self.reset()
        self.start_time = time.time()

    def get_elapsed(self):
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def get_phase(self):
        elapsed = self.get_elapsed()
        if elapsed < STABILIZATION_TIME:
            return 1
        elif elapsed < SESSION_DURATION:
            return 2
        else:
            return 3  # Final

    def is_session_active(self):
        return self.start_time is not None and not self.is_finalized

    def is_session_complete(self):
        return self.get_elapsed() >= SESSION_DURATION

    def add_frame_features(self, feature_dict):
        """
        Add features from a processed frame.
        Phase 1: ignore (caller uses the prediction for UI only)
        Phase 2: accumulate into array
        """
        self.frame_count += 1
        phase = self.get_phase()

        if phase == 2:
            self.phase2_features.append(feature_dict)
            self.phase2_frame_count += 1

        return phase

    def compute_final_verdict(self, predictor):
        """
        At t=15s: Average all Phase 2 features, predict once.
        Returns (hg_estimate, num_frames_used) or (None, 0) if insufficient data.
        """
        if len(self.phase2_features) == 0:
            return None, 0

        # Mean-average each of the 142 features
        averaged = {}
        for name in self.feature_names:
            values = [f.get(name, 0) for f in self.phase2_features]
            averaged[name] = float(np.mean(values))

        hg = predictor.predict(averaged)
        self.final_result = hg
        self.is_finalized = True

        return hg, len(self.phase2_features)


# =============================================================================
# UNIFIED INFERENCE ENGINE
# =============================================================================
class InferenceEngine:
    """
    Main entry point. Combines YOLO detection, feature extraction,
    XGBoost prediction, and session management.
    """

    def __init__(self):
        self.detector = YOLODetector()
        self.predictor = HemoglobinPredictor()
        self.sessions = {}  # client_id → SessionManager

    def get_session(self, client_id):
        if client_id not in self.sessions:
            self.sessions[client_id] = SessionManager()
        return self.sessions[client_id]

    def start_session(self, client_id):
        session = self.get_session(client_id)
        session.start()
        return {'status': 'session_started', 'duration': SESSION_DURATION}

    def process_frame(self, client_id, img_rgb):
        """
        Process a single frame from a client.

        Returns dict with:
          - status: 'ok', 'no_blood', 'only_N_patches', etc.
          - phase: 1 / 2 / 3
          - elapsed: seconds since session start
          - hg_preview: per-frame prediction (phase 1 & 2, for UI)
          - hg_final: final medical estimate (phase 3 only)
          - blood_detected: bool
          - patches_found: int
          - is_final: bool
          - frames_used: int (phase 2 frames for final average)
        """
        session = self.get_session(client_id)
        elapsed = session.get_elapsed()
        phase = session.get_phase()

        result = {
            'status': 'processing',
            'phase': phase,
            'elapsed': round(elapsed, 2),
            'hg_preview': None,
            'hg_final': None,
            'blood_detected': False,
            'patches_found': 0,
            'is_final': False,
            'frame_count': session.frame_count,
            'phase2_frames': session.phase2_frame_count,
        }

        # If session already finalized
        if session.is_finalized:
            result['status'] = 'complete'
            result['is_final'] = True
            result['hg_final'] = session.final_result
            return result

        # Check if time to finalize
        if phase == 3:
            hg, frames_used = session.compute_final_verdict(self.predictor)
            result['status'] = 'final_verdict'
            result['is_final'] = True
            result['hg_final'] = round(hg, 1) if hg else None
            result['frames_used'] = frames_used
            return result

        # --- YOLO Detection ---
        blood_box, sorted_patches, status = self.detector.detect(img_rgb)

        if status != 'ok' or blood_box is None:
            result['status'] = status
            return result

        result['blood_detected'] = True
        result['patches_found'] = len(sorted_patches) if sorted_patches else 0

        # --- Extract 142 features ---
        features = extract_all_features(blood_box, sorted_patches)

        # --- Phase-aware processing ---
        current_phase = session.add_frame_features(features)

        # Preview prediction for UI (both phases)
        try:
            hg_preview = self.predictor.predict(features)
            result['hg_preview'] = round(hg_preview, 1)
        except Exception:
            result['hg_preview'] = None

        result['status'] = 'ok'
        result['phase'] = current_phase

        # Detection coordinates for overlay
        result['detections'] = {
            'blood': {
                'xyxy': blood_box['xyxy'],
                'conf': round(blood_box['conf'], 2),
            },
            'patches': [
                {
                    'xyxy': p['xyxy'],
                    'label': f'P{i+1}' + (' (W)' if i == 0 else ''),
                }
                for i, p in enumerate(sorted_patches)
            ]
        }

        return result

    def finalize_session(self, client_id):
        """
        Explicitly compute the final verdict.
        Called when the client's 5s timer expires, regardless of server clock.
        This solves the race condition where client and server clocks differ.
        """
        session = self.get_session(client_id)

        if session.is_finalized:
            return {
                'status': 'complete',
                'is_final': True,
                'hg_final': session.final_result,
                'phase': 3,
                'elapsed': round(session.get_elapsed(), 2),
                'frame_count': session.frame_count,
                'phase2_frames': session.phase2_frame_count,
                'frames_used': session.phase2_frame_count,
            }

        hg, frames_used = session.compute_final_verdict(self.predictor)
        return {
            'status': 'final_verdict',
            'is_final': True,
            'hg_final': round(hg, 1) if hg else None,
            'phase': 3,
            'elapsed': round(session.get_elapsed(), 2),
            'frame_count': session.frame_count,
            'phase2_frames': session.phase2_frame_count,
            'frames_used': frames_used,
        }

    def process_offline_frames(self, frames_rgb):
        """
        Process a list of offline frames (e.g. from an uploaded video/folder).
        Simulates Phase 2 accumulation and Final Verdict.
        """
        phase2_features = []
        
        for img_rgb in frames_rgb:
            blood_box, sorted_patches, status = self.detector.detect(img_rgb)
            if status == 'ok' and blood_box is not None:
                features = extract_all_features(blood_box, sorted_patches)
                phase2_features.append(features)
        
        if len(phase2_features) == 0:
            return {'status': 'error', 'message': 'No valid frames with blood/patches detected.'}
            
        # Mean-average each of the 142 features
        averaged = {}
        for name in self.predictor.feature_names:
            values = [f.get(name, 0) for f in phase2_features]
            averaged[name] = float(np.mean(values))
            
        try:
            hg = self.predictor.predict(averaged)
        except Exception as e:
            return {'status': 'error', 'message': f'Prediction error: {str(e)}'}
            
        return {
            'status': 'success',
            'hg_final': round(hg, 1),
            'frames_used': len(phase2_features),
            'total_extracted': len(frames_rgb)
        }

    def end_session(self, client_id):
        if client_id in self.sessions:
            del self.sessions[client_id]
