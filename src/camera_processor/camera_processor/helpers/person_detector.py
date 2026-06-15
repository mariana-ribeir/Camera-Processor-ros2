import cv2
import numpy as np
from collections import deque

# Detection/tracking configuration
DEFAULT_CONF = 0.40
IOU_THRESHOLD = 0.40
MAX_TRACK_AGE = 30
COLOR_HISTORY_LEN = 50
COLOR_SWITCH_CONFIRM_FRAMES = 25
COLOR_LOCK_MIN_STREAK = 8
COLOR_MIN_COVERAGE = 0.15
TORSO_CENTER_FRAC = 0.70
KP_CONF_MIN = 0.50

KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_HIP = 11
KP_R_HIP = 12

KP_SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (11, 12),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2),
    (1, 3), (2, 4),
]

# Fixed person IDs by shirt color
COLOR_TO_ID = {
    "RED": 1,
    "ORANGE": 2,
    "YELLOW": 3,
    "GREEN": 3,
    "BLUE": 5,
}

# HSV ranges for shirt-color detection
HSV_RANGES = {
    "RED": [((0, 120, 80), (5, 255, 255)), ((170, 120, 80), (180, 255, 255))],
    "ORANGE": [((6, 120, 120), (18, 255, 255))],
    "YELLOW": [((20, 50, 50), (85, 255, 255))],
    "GREEN": [((20, 50, 50), (85, 255, 255))],
    "BLUE": [((100, 120, 70), (128, 255, 255))],
}

VIS_COLORS = {
    "RED": (0, 0, 255),
    "ORANGE": (0, 128, 255),
    "YELLOW": (0, 255, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "UNKNOWN": (140, 140, 140),
}


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def extract_upper_torso_roi(frame: np.ndarray, bbox: tuple, center_frac: float, keypoints=None):
    """
    Extract a central upper-body ROI from a person bounding box.

    This works without keypoints/segmentation masks and is robust for ONNX
    detector-only outputs.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 4 or bh <= 4:
        return None, None

    tx1 = ty1 = tx2 = ty2 = None

    # Preferred ROI: torso box derived from shoulder/hip keypoints.
    if keypoints is not None and len(keypoints) >= 13:
        torso_kp_indices = [KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP]
        valid_pts = [keypoints[i] for i in torso_kp_indices if keypoints[i][2] >= KP_CONF_MIN]
        if len(valid_pts) >= 2:
            pts = np.array([[k[0], k[1]] for k in valid_pts], dtype=np.float32)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            tx1, ty1, tx2, ty2 = float(x_min), float(y_min), float(x_max), float(y_max)

    # Fallback ROI: upper-mid bbox when keypoints are missing/unreliable.
    if tx1 is None:
        tx1 = x1 + int(0.20 * bw)
        tx2 = x2 - int(0.20 * bw)
        ty1 = y1 + int(0.12 * bh)
        ty2 = y1 + int(0.55 * bh)

    # Keep a centered fraction to reduce background/arms influence.
    cx = (tx1 + tx2) / 2.0
    cy = (ty1 + ty2) / 2.0
    x_half = max(((tx2 - tx1) * center_frac) / 2.0, 8)
    y_half = max(((ty2 - ty1) * center_frac) / 2.0, 8)

    rx1 = int(np.clip(cx - x_half, 0, w - 1))
    ry1 = int(np.clip(cy - y_half, 0, h - 1))
    rx2 = int(np.clip(cx + x_half, 0, w - 1))
    ry2 = int(np.clip(cy + y_half, 0, h - 1))

    if rx2 <= rx1 or ry2 <= ry1:
        return None, None

    roi = frame[ry1:ry2, rx1:rx2]
    return (roi if roi.size > 0 else None), (rx1, ry1, rx2, ry2)


def detect_shirt_color(roi: np.ndarray) -> str:
    """Detect dominant shirt color in torso ROI using HSV masks."""
    if roi is None or roi.size == 0:
        return "UNKNOWN"

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]

    pixel_counts = {}
    for color_name, ranges in HSV_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(
                hsv,
                np.array(lo, dtype=np.uint8),
                np.array(hi, dtype=np.uint8),
            )
        pixel_counts[color_name] = int(np.count_nonzero(mask))

    best_color = max(pixel_counts, key=pixel_counts.get)
    if pixel_counts[best_color] < total * COLOR_MIN_COVERAGE:
        return "UNKNOWN"
    return best_color


class ColorStabilizer:
    """Temporal voting to stabilize color labels per track."""

    def __init__(self, history_len: int = COLOR_HISTORY_LEN):
        self._history_len = history_len
        self._histories = {}

    def update(self, track_id: int, raw_color: str) -> str:
        if track_id not in self._histories:
            self._histories[track_id] = deque(maxlen=self._history_len)
        self._histories[track_id].append(raw_color)

        counts = {}
        for color in self._histories[track_id]:
            counts[color] = counts.get(color, 0) + 1

        valid = {k: v for k, v in counts.items() if k != "UNKNOWN"}
        return max(valid, key=valid.get) if valid else "UNKNOWN"

    def force_color(self, track_id: int, color: str) -> str:
        self._histories[track_id] = deque([color], maxlen=self._history_len)
        return color

    def remove_stale(self, active_ids: set) -> None:
        for track_id in list(self._histories.keys()):
            if track_id not in active_ids:
                del self._histories[track_id]


class SimpleTracker:
    """IoU-based lightweight tracker with per-track color memory."""

    def __init__(self):
        self._tracks = {}
        self._next_id = 1

    def update(self, detections: list, frame_idx: int) -> list:
        used_tracks = set()
        results = []

        for det in detections:
            best_tid = None
            best_iou = 0.0

            for tid, track in self._tracks.items():
                if tid in used_tracks:
                    continue
                iou = compute_iou(det["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= IOU_THRESHOLD:
                prev_best_color = self._tracks[best_tid].get("best_color", "UNKNOWN")
                self._tracks[best_tid]["bbox"] = det["bbox"]
                self._tracks[best_tid]["last_seen"] = frame_idx
                used_tracks.add(best_tid)
                results.append((det, best_tid, prev_best_color, best_iou))
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "bbox": det["bbox"],
                    "last_seen": frame_idx,
                    "best_color": "UNKNOWN",
                    "pending_color": None,
                    "pending_count": 0,
                    "best_color_streak": 0,
                }
                used_tracks.add(tid)
                results.append((det, tid, "UNKNOWN", 0.0))

        for tid in list(self._tracks.keys()):
            if frame_idx - self._tracks[tid]["last_seen"] > MAX_TRACK_AGE:
                del self._tracks[tid]

        return results

    def set_track_color(self, tid: int, color: str) -> None:
        if tid not in self._tracks:
            return
        prev = self._tracks[tid].get("best_color", "UNKNOWN")
        if color == prev:
            self._tracks[tid]["best_color_streak"] = self._tracks[tid].get("best_color_streak", 0) + 1
        else:
            self._tracks[tid]["best_color_streak"] = 1 if color in COLOR_TO_ID else 0
        self._tracks[tid]["best_color"] = color

    def apply_color_memory(self, tid: int, raw_color: str, matched_iou: float) -> str:
        track = self._tracks.get(tid)
        if track is None:
            return raw_color

        prev_color = track.get("best_color", "UNKNOWN")
        prev_streak = track.get("best_color_streak", 0)

        if prev_color not in COLOR_TO_ID:
            track["pending_color"] = None
            track["pending_count"] = 0
            return raw_color

        if matched_iou < IOU_THRESHOLD:
            track["pending_color"] = None
            track["pending_count"] = 0
            return raw_color

        if raw_color == "UNKNOWN":
            return prev_color

        if raw_color == prev_color:
            track["pending_color"] = None
            track["pending_count"] = 0
            return prev_color

        pending_color = track.get("pending_color")
        pending_count = track.get("pending_count", 0)
        if raw_color == pending_color:
            pending_count += 1
        else:
            pending_color = raw_color
            pending_count = 1

        track["pending_color"] = pending_color
        track["pending_count"] = pending_count

        required = COLOR_SWITCH_CONFIRM_FRAMES
        if prev_streak >= COLOR_LOCK_MIN_STREAK:
            required += 5

        if pending_count >= required:
            track["pending_color"] = None
            track["pending_count"] = 0
            return raw_color

        return prev_color

    def active_ids(self) -> set:
        return set(self._tracks.keys())


def resolve_unique_ids(candidates: list) -> dict:
    """Ensure one fixed color-ID winner per color in the current frame."""
    by_color = {}
    for item in candidates:
        color = item["stable_color"]
        if color not in COLOR_TO_ID:
            continue
        x1, y1, x2, y2 = item["det"]["bbox"]
        conf = float(item["det"].get("conf", 0.0))
        area = max(1, (x2 - x1) * (y2 - y1))
        score = (conf, area)
        by_color.setdefault(color, []).append((score, item["tid"]))

    final_by_tid = {item["tid"]: item["stable_color"] for item in candidates}

    for color, scored_tids in by_color.items():
        scored_tids.sort(key=lambda x: x[0], reverse=True)
        winner_tid = scored_tids[0][1]
        for _, tid in scored_tids[1:]:
            if tid != winner_tid:
                final_by_tid[tid] = "UNKNOWN"

    return final_by_tid


def draw_annotations(frame: np.ndarray, det: dict, label_color: str, torso_rect) -> None:
    """Draw bbox, torso ROI, and final ID-color label on frame."""
    x1, y1, x2, y2 = det["bbox"]
    color = VIS_COLORS.get(label_color, VIS_COLORS["UNKNOWN"])

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    keypoints = det.get("keypoints")
    if keypoints is not None and len(keypoints) >= 17:
        for a, b in KP_SKELETON:
            if keypoints[a][2] >= KP_CONF_MIN and keypoints[b][2] >= KP_CONF_MIN:
                p1 = (int(keypoints[a][0]), int(keypoints[a][1]))
                p2 = (int(keypoints[b][0]), int(keypoints[b][1]))
                cv2.line(frame, p1, p2, color, 2)

        for k in keypoints:
            if k[2] >= KP_CONF_MIN:
                cv2.circle(frame, (int(k[0]), int(k[1])), 4, (255, 255, 255), -1)
                cv2.circle(frame, (int(k[0]), int(k[1])), 4, color, 1)

    if torso_rect is not None:
        tx1, ty1, tx2, ty2 = torso_rect
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 220, 220), 1)

    pid = COLOR_TO_ID.get(label_color)
    conf = det.get("conf", 0.0)
    base = f"ID {pid} - {label_color}" if pid is not None else label_color
    label = f"{base} YOLO: {conf * 100:.0f}%"
    y_lbl = max(y1 - 8, 12)
    cv2.putText(frame, label, (x1, y_lbl), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 3)
    cv2.putText(frame, label, (x1, y_lbl), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)


_TRACKER = SimpleTracker()
_STABILIZER = ColorStabilizer()
_FRAME_INDEX = 0


def reset_person_database() -> None:
    """Compatibility entrypoint used by previous pipeline."""
    global _TRACKER, _STABILIZER, _FRAME_INDEX
    _TRACKER = SimpleTracker()
    _STABILIZER = ColorStabilizer()
    _FRAME_INDEX = 0


def person_process_frame(frame, model):
    """
    Process a frame with YOLO ONNX detections and color-based identity assignment.

    Returns:
        annotated_frame, people_detected, people_count, detections
        detections item format: (id, x, y, w, h, confidence)
    """
    global _FRAME_INDEX
    _FRAME_INDEX += 1

    results = model(frame, verbose=False, conf=DEFAULT_CONF, classes=[0])
    annotated_frame = frame.copy()
    detections = []

    keypoints_data = None
    if results and len(results) > 0 and getattr(results[0], "keypoints", None) is not None:
        if getattr(results[0].keypoints, "data", None) is not None:
            keypoints_data = results[0].keypoints.data

    if results and results[0].boxes is not None:
        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 <= x1 or y2 <= y1:
                continue

            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            kpts = None
            if keypoints_data is not None and idx < len(keypoints_data):
                kpts_t = keypoints_data[idx]
                if hasattr(kpts_t, "cpu"):
                    kpts = kpts_t.cpu().numpy().astype(np.float32)
                else:
                    kpts = np.array(kpts_t, dtype=np.float32)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "keypoints": kpts,
            })

    tracked = _TRACKER.update(detections, _FRAME_INDEX)
    _STABILIZER.remove_stale(_TRACKER.active_ids())

    candidates = []
    for det, tid, prev_best_color, matched_iou in tracked:
        roi, torso_rect = extract_upper_torso_roi(frame, det["bbox"], TORSO_CENTER_FRAC, det.get("keypoints"))
        detected_color = detect_shirt_color(roi)
        raw_color = _TRACKER.apply_color_memory(tid, detected_color, matched_iou)

        confirmed_color_change = (
            prev_best_color in COLOR_TO_ID
            and raw_color in COLOR_TO_ID
            and raw_color != prev_best_color
        )

        if confirmed_color_change:
            stable_color = _STABILIZER.force_color(tid, raw_color)
        else:
            stable_color = _STABILIZER.update(tid, raw_color)

        _TRACKER.set_track_color(tid, stable_color)
        candidates.append({
            "det": det,
            "tid": tid,
            "torso_rect": torso_rect,
            "stable_color": stable_color,
        })

    final_by_tid = resolve_unique_ids(candidates)
    ros_detections = []

    for item in candidates:
        det = item["det"]
        x1, y1, x2, y2 = det["bbox"]
        w = x2 - x1
        h = y2 - y1
        confidence = float(det.get("conf", 0.0))

        final_color = final_by_tid.get(item["tid"], "UNKNOWN")
        draw_annotations(annotated_frame, det, final_color, item["torso_rect"])

        # Keep fixed ID by color when possible. If unknown, publish sentinel 0.
        person_id = COLOR_TO_ID.get(final_color, 0)
        ros_detections.append((person_id, x1, y1, w, h, confidence))

    people_detected = len(ros_detections) > 0
    people_count = len(ros_detections)
    return annotated_frame, people_detected, people_count, ros_detections
