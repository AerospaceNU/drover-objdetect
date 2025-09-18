import cv2 as cv
import numpy as np
from typing import Tuple
import os
import json


class Detection:
    def __init__(
        self, weights: str = "threshold.json", kernel: Tuple[int, int] = (7, 7)
    ):
        self.low_H = 0
        self.low_S = 0
        self.low_V = 0
        self.high_H = 0
        self.high_S = 0
        self.high_V = 0

        self.max_val = 255

        self.weights = weights
        if weights is not None:
            self.load_threshold()

        self.kernel = kernel
        self.fgbg = cv.createBackgroundSubtractorMOG2()

    def save_threshold(self):
        dir_name = os.path.dirname(self.weights)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        data = {
            "low_H": self.low_H,
            "low_S": self.low_S,
            "low_V": self.low_V,
            "high_H": self.high_H,
            "high_S": self.high_S,
            "high_V": self.high_V,
        }

        with open(self.weights, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return self.weights

    def load_threshold(self):
        if not os.path.exists(self.weights):
            self.save_threshold(self.weights)

        with open(self.weights, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.low_H = data["low_H"]
        self.low_S = data["low_S"]
        self.low_V = data["low_V"]
        self.high_H = data["high_H"]
        self.high_S = data["high_S"]
        self.high_V = data["high_V"]

    def filter_hsv(self, frame, avg=None):
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        if avg is None:
            avg = cv.mean(frame_HSV)
        color_filter = cv.bitwise_not(
            cv.inRange(
                frame_HSV,
                (max(avg[0] - self.low_H, 0), max(avg[1] - self.low_S, 0), 0),
                (min(avg[0] + self.high_H, 180), min(avg[1] + self.high_S, 255), 255),
            )
        )
        value_filter = cv.inRange(
            frame_HSV,
            (0, 0, max(avg[2] - self.low_V, 0)),
            (180, 255, min(avg[2] + self.high_V, 255)),
        )
        return cv.bitwise_and(color_filter, value_filter), avg

    def detect(self, frame: np.ndarray):
        frame_blurred = cv.GaussianBlur(frame, self.kernel, 0)
        fgmask = self.fgbg.apply(frame_blurred)
        frame_copy = frame.copy()
        fgmask_copy = fgmask.copy()

        frame_threshold, mean = self.filter_hsv(frame_blurred)
        contours, h = cv.findContours(
            fgmask_copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, width, height = cv.boundingRect(contour)
            if width * height < 20:
                fgmask_copy[y : y + height, x : x + width] = 0
                continue

            filtered, _ = self.filter_hsv(
                frame_blurred[y : y + height, x : x + width], mean
            )
            if (
                np.count_nonzero(filtered) / (np.multiply.reduce(filtered.shape) / 255)
                < 0.2
            ):
                fgmask_copy[y : y + height, x : x + width] = 0
                continue

            fgmask_copy[y : y + height, x : x + width] = cv.bitwise_and(
                fgmask[y : y + height, x : x + width], filtered
            )

            frame_copy = cv.rectangle(
                frame_copy, (x, y), (x + width, y + height), (0, 255, 0), 2
            )
            
            fgmask_copy = cv.rectangle(
                fgmask_copy, (x, y), (x + width, y + height), 255, 2
            )

        return frame_copy, fgmask_copy

    def set_attr(self, name: str, val: float):
        val = min(self.max_val, max(0, val))
        setattr(self, name, val)
        return getattr(self, name)
