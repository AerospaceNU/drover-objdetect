import cv2 as cv
import numpy as np
from typing import Union


class VideoFeed:
    def __init__(self, source: Union[int, str]):
        if isinstance(source, int):
            self.feed = cv.VideoCapture(source)
            if not self.feed.isOpened():
                raise RuntimeError(f"Could not open camera device {source}")
        elif isinstance(source, str):
            self.feed = cv.VideoCapture(source)
            if not self.feed.isOpened():
                raise FileNotFoundError(f"Could not open designated mp4")
        else:
            raise ValueError(
                "Invalid source type. It should either be a String, pointing to an mp4, or an integer"
            )

        self.last_frame = None
        self.is_pausing = False
        self.stop_reading = False

    def read(self):
        if not self.stop_reading:
            ret, frame = self.feed.read()
            if ret:
                self.last_frame = frame
            else:
                return None
        return self.last_frame

    def toggle_pause(self):
        self.stop_reading = not self.stop_reading

    def release(self):
        if self.feed is not None:
            self.feed.release()
