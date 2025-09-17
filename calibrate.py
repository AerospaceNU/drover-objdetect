import cv2 as cv
import numpy as np
import json
import os

max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = 0
high_S = 0
high_V = 0
window_capture_name = "Video Capture"
window_detection_name = "Object Detection"
low_H_name = "Lower Bound Offset H"
low_S_name = "Lower Bound Offset S"
low_V_name = "Lower Bound Offset V"
high_H_name = "Upper Bound Offset H"
high_S_name = "Upper Bound Offset S"
high_V_name = "Upper Bound Offset V"

DEFAULT_PATH = "threshold.json"


def save_threshold(path: str = DEFAULT_PATH):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    data = {
        "low_H": low_H,
        "low_S": low_S,
        "low_V": low_V,
        "high_H": high_H,
        "high_S": high_S,
        "high_V": high_V,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path


def load_threshold(path: str = DEFAULT_PATH):
    global low_H, low_S, low_V, high_H, high_S, high_V

    if not os.path.exists(path):
        save_threshold(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    low_H = data["low_H"]
    low_S = data["low_S"]
    low_V = data["low_V"]
    high_H = data["high_H"]
    high_S = data["high_S"]
    high_V = data["high_V"]


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)


load_threshold()
cap = cv.VideoCapture("testing.mp4")
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(
    low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar
)
cv.createTrackbar(
    high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar
)
cv.createTrackbar(
    low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar
)
cv.createTrackbar(
    high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar
)
cv.createTrackbar(
    low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar
)
cv.createTrackbar(
    high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar
)


# filter out average color
# dont really filter saturation
# filter for higher than average value


def filter_hsv(frame, avg=None):
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    if avg is None:
        avg = cv.mean(frame_HSV)
    color_filter = cv.bitwise_not(
        cv.inRange(
            frame_HSV,
            (max(avg[0] - low_H, 0), max(avg[1] - low_S, 0), 0),
            (min(avg[0] + high_H, 180), min(avg[1] + high_S, 255), 255),
        )
    )
    value_filter = cv.inRange(
        frame_HSV, (0, 0, max(avg[2] - low_V, 0)), (180, 255, min(avg[2] + high_V, 255))
    )
    return cv.bitwise_and(color_filter, value_filter), avg


fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

stop_reading = False
frame = None
while True:
    if not stop_reading:
        ret, frame = cap.read()
        if not ret:
            break
        frame_blurred = cv.GaussianBlur(frame, (3, 3), 0)
        fgmask = fgbg.apply(frame_blurred)
    frame_copy = frame.copy()
    fgmask_copy = fgmask.copy()

    frame_threshold, mean = filter_hsv(frame_blurred)
    contours, h = cv.findContours(fgmask_copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, width, height = cv.boundingRect(contour)
        if width * height < 20:
            fgmask_copy[y : y + height, x : x + width] = 0
            continue
        print(width, height)
        filtered, _ = filter_hsv(frame_blurred[y : y + height, x : x + width], mean)
        if (
            np.count_nonzero(filtered) / (np.multiply.reduce(filtered.shape) / 255)
            < 0.2
        ):
            fgmask_copy[y : y + height, x : x + width] = 0
            continue
        fgmask_copy[y : y + height, x : x + width] = cv.bitwise_and(
            fgmask[y : y + height, x : x + width], filtered
        )
        fgmask_copy = cv.rectangle(
            fgmask_copy, (x, y), (x + width, y + height), (0, 255, 0), 2
        )

    # cv.drawContours(frame_copy, contours, -1, (0, 255, 0), 3)

    cv.imshow(window_capture_name, frame_copy)
    cv.imshow(window_detection_name, cv.resize(fgmask_copy, (960, 540)))
    cv.imshow("win", fgmask_copy)

    key = cv.waitKey(30)
    if key == ord("q") or key == 27:
        break
    if key == ord(" "):
        stop_reading = not stop_reading

save_threshold()
