from typing import List, Tuple, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment
import math


class KalmanFilter:

    def __init__(self, initial_state: np.ndarray, dt: float = 1.0):
        self.dt = dt

        self.F = np.array(
            [
                [1, 0, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, 0, dt],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )

        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.x = initial_state

        self.P = np.diag([50.0, 50.0, 1e4, 10.0, 500.0, 500.0, 1e3]).astype(float)

        self.Q = np.diag([1.0, 1.0, 10.0, 0.01, 1.0, 1.0, 10.0]).astype(float)

        self.R = np.diag([10.0, 10.0, 100.0, 0.1]).astype(float)

        self.I = np.eye(7)

        self.s_min = 1e-3
        self.r_min, self.r_max = 0.2, 5.0

    def _sanitize(self):
        if self.x[2, 0] < self.s_min:
            self.x[2, 0] = self.s_min
            self.x[6, 0] = 0.0
        self.x[3, 0] = float(np.clip(self.x[3, 0], self.r_min, self.r_max))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self._sanitize()
        return self.x

    def update(self, measurement: np.ndarray):
        s_meas = float(measurement[2])
        self.R[2, 2] = max(1e2, (0.05 * max(1.0, s_meas)) ** 2)

        y = measurement - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        self.P = (self.I - (K @ self.H)) @ self.P
        self._sanitize()
        return self.x

    def get_state(self) -> np.ndarray:
        return self.x


def bbox_to_state(bbox: np.ndarray):
    measurement = bbox_to_measurement(bbox)
    return np.vstack([measurement, np.zeros((3, 1))], dtype=np.float32)


def bbox_to_measurement(bbox: np.ndarray):
    x1, y1, x2, y2 = map(float, bbox)
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    u = x1 + w / 2.0
    v = y1 + h / 2.0
    s = w * h
    r = w / h
    return np.array([[u], [v], [s], [r]], dtype=float)


def state_to_bbox(state: np.ndarray):
    u, v, s, r = float(state[0]), float(state[1]), float(state[2]), float(state[3])
    val = r * s
    if val <= 0:
        val = 1e-6
    w = math.sqrt(val)
    h = s / w
    x1 = u - w / 2
    x2 = u + w / 2
    y1 = v - h / 2
    y2 = v + h / 2
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou(bbox_1: np.ndarray, bbox_2: np.ndarray):
    x11, y11, x12, y12 = (
        float(bbox_1[0]),
        float(bbox_1[1]),
        float(bbox_1[2]),
        float(bbox_1[3]),
    )

    x21, y21, x22, y22 = (
        float(bbox_2[0]),
        float(bbox_2[1]),
        float(bbox_2[2]),
        float(bbox_2[3]),
    )

    inter_w = max(0, min(x12, x22) - max(x11, x21))
    inter_h = max(0, min(y12, y22) - max(y11, y21))
    inter_area = inter_h * inter_w

    total_area = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21) - inter_area

    if total_area <= 0:
        return 0.0

    iou = inter_area / total_area
    return max(0.0, min(1.0, iou))


class Track:
    def __init__(self, initial_payload: dict, track_id: int):
        self.track_id = track_id
        self.filter = KalmanFilter(initial_state=bbox_to_state(initial_payload["bbox"]))

        self.payload = dict(initial_payload)
        self.payload["track_id"] = self.track_id
        self.payload["kalman"] = False

        self.time_since_update = 0
        self.hit_streak = 1
        self.age = 1

    def predict(self, frame_idx: int):
        state = self.filter.get_state()
        bbox_pred = state_to_bbox(state).tolist()
        u, v = float(state[0]), float(state[1])

        payload = dict(self.payload)
        payload.update(
            {
                "frame_idx": frame_idx,
                "center": (u, v),
                "bbox": bbox_pred,
                "conf": None,
                "rel_depth_value": None,
                "ts_detect": None,
                "ts_depth": None,
            }
        )
        payload["kalman"] = True
        return payload

    def update(self, new_payload: dict):
        self.time_since_update = 0
        self.hit_streak += 1
        z = bbox_to_measurement(new_payload["bbox"])
        self.filter.update(z)

        self.payload = dict(new_payload)
        self.payload["track_id"] = self.track_id
        self.payload["kalman"] = False
        return dict(self.payload)

    def get_track_id(self):
        return self.track_id

    def get_bbox(self):
        return state_to_bbox(self.filter.get_state())

    def get_state(self):
        return self.filter.get_state()

    def kf_predict_only(self):
        self.filter.predict()


class SORTTrackManager:
    def __init__(
        self,
        max_age: int = 10,
        min_age: int = 5,
        iou_threshold: float = 0.1,
    ):
        self.max_age = max_age
        self.min_age = min_age
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.curr_track_id = 0

    def step(self, detections: List[Dict], frame_idx: int) -> List[Dict]:
        output = []
        matches = []
        unmatched_tracks = []
        unmatched_dets = []

        if not self.tracks:
            for det in detections:
                new_track = Track(det, self.curr_track_id)
                self.curr_track_id += 1
                self.tracks.append(new_track)
                output.append(dict(new_track.payload))
            return output

        for t in self.tracks:
            t.kf_predict_only()
            t.age += 1
            t.time_since_update += 1

        if not detections:
            for track in self.tracks:
                if track.time_since_update <= self.max_age:
                    state = track.get_state()
                    payload = dict(track.payload)
                    payload.update(
                        {
                            "frame_idx": frame_idx,
                            "center": (float(state[0]), float(state[1])),
                            "bbox": state_to_bbox(state).tolist(),
                            "conf": None,
                            "rel_depth_value": None,
                            "ts_detect": None,
                            "ts_depth": None,
                            "kalman": True,
                        }
                    )
                    output.append(payload)
            self.tracks = [
                t for t in self.tracks if t.time_since_update <= self.max_age
            ]
            return output

        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for t_idx, track in enumerate(self.tracks):
            pred_bbox = track.get_bbox()
            for d_idx, detection in enumerate(detections):
                iou_matrix[t_idx, d_idx] = iou(pred_bbox, detection["bbox"])

        rows, cols = linear_sum_assignment(-iou_matrix)
        assigned_tracks, assigned_detection = set(), set()
        for r, c in zip(rows, cols):
            if iou_matrix[r, c] > self.iou_threshold:
                matches.append((r, c))
                assigned_tracks.add(r)
                assigned_detection.add(c)

        for t_idx in range(len(self.tracks)):
            if t_idx not in assigned_tracks:
                unmatched_tracks.append(t_idx)

        for d_idx in range(len(detections)):
            if d_idx not in assigned_detection:
                unmatched_dets.append(d_idx)

        for t_idx, d_idx in matches:
            payload = self.tracks[t_idx].update(detections[d_idx])
            self.tracks[t_idx].time_since_update = 0
            self.tracks[t_idx].age = 0
            output.append(payload)

        for idx in unmatched_tracks:
            track = self.tracks[idx]
            if track.time_since_update <= self.max_age:
                state = track.get_state()
                payload = dict(track.payload)
                payload.update(
                    {
                        "frame_idx": frame_idx,
                        "center": (float(state[0]), float(state[1])),
                        "bbox": state_to_bbox(state).tolist(),
                        "conf": None,
                        "rel_depth_value": None,
                        "ts_detect": None,
                        "ts_depth": None,
                        "kalman": True,
                    }
                )
                output.append(payload)

        for idx in unmatched_dets:
            det = detections[idx]

            if det.get("conf", 1.0) < 0.75:
                continue

            bbox_det = np.array(det["bbox"])
            too_close = False
            for out in output:
                if iou(bbox_det, np.array(out["bbox"])) > 0.65:
                    too_close = True
                    break

            if too_close:
                continue

            new_track = Track(det, self.curr_track_id)
            self.tracks.append(new_track)
            output.append(dict(new_track.payload))
            self.curr_track_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return output
