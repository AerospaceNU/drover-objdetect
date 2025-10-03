import cv2 as cv
import numpy as np
from typing import Union


class VideoFeed:
    """
    The VideoFeed class handles the video input, whether from a camera or a file.
    It provides methods to read frames, toggle pause, and release the video source.
    """

    def __init__(self, source: Union[int, str]):
        """
        Initializes the VideoFeed object.

        Args:
            source (Union[int, str]): The video source, which can be a camera index (int)
                                      for device camera feed to or a file path for video.
        """
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
        self.frame_idx = 0

    def read(self):
        """
        Reads a frame from the video feed.

        Returns:
            The video frame, or None if the feed has ended.
        """
        if not self.stop_reading:
            ret, frame = self.feed.read()
            if ret:
                self.last_frame = frame
                self.frame_idx += 1
            else:
                return None
        return self.last_frame

    def toggle_pause(self):
        """Toggles the pause state of the video feed."""
        self.stop_reading = not self.stop_reading

    def release(self):
        """Releases the video capture object."""
        if self.feed is not None:
            self.feed.release()
