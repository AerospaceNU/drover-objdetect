import cv2 as cv
import numpy as np
from detection import Detection
from videoFeed import VideoFeed
from typing import Union
from enum import Enum
import os
from datetime import datetime


class VideoFeedOptions(Enum):
    RAW = 1
    RAW_ANNOTATED = 2
    FGMASK = 3


class Display:
    """
    The Display class manages the GUI, including video display, trackbars for parameter
    tuning, and handling user keyboard inputs.
    """

    def __init__(self, source: Union[int, str], windowName: str, **kwargs):
        """
        Initializes the Display object.

        Args:
            source (Union[int, str]): The video source (camera index or file path).
            windowName (str): The name of the main display window.
            **kwargs: Additional arguments for the Detection class.
        """
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

        self._init_video_writers()

    def _init_video_writers(self):
        """
        Initialize video writers for each frame type. Video types includes raw, annotated, annotated foreground.
        Defaults output directory to output_videos
        """

        output_dir = "output_videos"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        ret, sample_frame = self.video.feed.read()
        if not ret:
            raise RuntimeError(
                "Could not read sample frame to determine video properties"
            )

        self.video.feed.set(cv.CAP_PROP_POS_FRAMES, 0)

        height, width = sample_frame.shape[:2]
        fps = self.video.feed.get(cv.CAP_PROP_FPS) or 30.0

        fourcc = cv.VideoWriter_fourcc(*"XVID")

        self.video_writers = {
            "raw": cv.VideoWriter(
                os.path.join(output_dir, f"raw_{timestamp}.mp4"),
                fourcc,
                fps,
                (width, height),
            ),
            "annotated": cv.VideoWriter(
                os.path.join(output_dir, f"annotated_{timestamp}.mp4"),
                fourcc,
                fps,
                (width, height),
            ),
            "fgmask": cv.VideoWriter(
                os.path.join(output_dir, f"fgmask_{timestamp}.mp4"),
                fourcc,
                fps,
                (width, height),
            ),
        }

        for name, writer in self.video_writers.items():
            if not writer.isOpened():
                print(f"Warning: Could not initialize video writer for {name}")

    def _write_frames(self, frame_raw, frame_annotated, fgmask):
        """Write frames to their corresponding video files."""
        if not self.stop_reading:
            if self.video_writers["raw"].isOpened():
                self.video_writers["raw"].write(frame_raw)

            if self.video_writers["annotated"].isOpened():
                self.video_writers["annotated"].write(frame_annotated)

            if self.video_writers["fgmask"].isOpened():
                fgmask_color = cv.cvtColor(fgmask, cv.COLOR_GRAY2BGR)
                self.video_writers["fgmask"].write(fgmask_color)

    def _release_video_writers(self):
        """Release all video writers and save the video files."""
        print("Saving video files...")
        for name, writer in self.video_writers.items():
            if writer.isOpened():
                writer.release()
                print(f"Saved {name} video")
        print("All video files saved successfully!")

    def update_param(self, name, val):
        """Update GUI param + Detection object."""
        self.params[name] = val
        self.detection.set_attr(name, val)

    def run(self):
        """Main loop: get frame, process it, show selected option."""
        if self.stop_reading:
            self.video.toggle_pause()

        frame_raw = self.video.read()
        if frame_raw is None:
            return

        frame_annotated, fgmask = self.detection.detect(frame_raw)

        self._write_frames(frame_raw, frame_annotated, fgmask)

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
            self._release_video_writers()
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
