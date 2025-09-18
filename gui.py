import cv2 as cv
import numpy as np
from detection import Detection
from videoFeed import VideoFeed
from typing import Union
from enum import Enum


class VideoFeedOptions(Enum):
    RAW = 1
    RAW_ANNOTATED = 2
    FGMASK = 3


class Display:
    def __init__(self, source: Union[int, str], windowName: str, **kwargs):
        assert isinstance(source, (int, str)), "source must be int or str"

        self.windowName = windowName
        self.feed_option = VideoFeedOptions.RAW
        self.stop_reading = False

        cv.namedWindow(self.windowName)
        cv.namedWindow("Controls")

        self.detection = Detection(**kwargs)
        self.video = VideoFeed(source)

        self.params = {
            "low_H": self.detection.low_H,
            "high_H": self.detection.high_H,
            "low_S": self.detection.low_S,
            "high_S": self.detection.high_S,
            "low_V": self.detection.low_V,
            "high_V": self.detection.high_V,
        }

        self.max_vals = {
            "low_H": 180,
            "high_H": 180,
            "low_S": 255,
            "high_S": 255,
            "low_V": 255,
            "high_V": 255,
        }

        for name, init in self.params.items():
            cv.createTrackbar(
                name,
                "Controls",
                init,
                self.max_vals[name],
                lambda x, n=name: self.update_param(n, x),
            )

    def update_param(self, name, val):
        """Update GUI param + Detection object"""
        self.params[name] = val
        self.detection.set_attr(name, val)

    def run(self):
        """Main loop: get frame, process it, show selected option"""
        if self.stop_reading:
            self.video.toggle_pause()

        frame_raw = self.video.read()
        if frame_raw is None:
            return

        frame_annotated, fgmask = self.detection.detect(frame_raw)

        if self.feed_option == VideoFeedOptions.RAW:
            cv.imshow(self.windowName, frame_raw)
        elif self.feed_option == VideoFeedOptions.RAW_ANNOTATED:
            cv.imshow(self.windowName, frame_annotated)
        elif self.feed_option == VideoFeedOptions.FGMASK:
            cv.imshow(self.windowName, fgmask)

    def wait_key(self, delay=30):
        """Keyboard interface for feed switching, pause, quit."""
        key = cv.waitKey(delay) & 0xFF
        if key == ord("q") or key == 27:
            self.detection.save_threshold()
            self.video.release()
            return False
        elif key in (ord("p"), ord(" ")):
            self.stop_reading = not self.stop_reading
        elif key == ord("r"):
            self.feed_option = VideoFeedOptions.RAW
        elif key == ord("a"):
            self.feed_option = VideoFeedOptions.RAW_ANNOTATED
        elif key == ord("f"):
            self.feed_option = VideoFeedOptions.FGMASK
        return True
