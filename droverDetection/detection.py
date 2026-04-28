import cv2 as cv
import numpy as np
from typing import Tuple
from droverDetection.sort_manager import SORTTrackManager
import os
import json


class Detection:
	"""
	The Detection class handles the object detection logic, including
	color filtering and background subtraction.
	"""

	def __init__(
			self, kernel: Tuple[int, int] = (3, 3)
	):
		"""
		Initializes the Detection object.

		Args:
			kernel (Tuple[int, int]): Kernel size for Gaussian blur.
		"""

		self.kernel = kernel
		self._zscore_bufs = None

	def _get_zscore_bufs(self, h, w):
		if self._zscore_bufs is None or self._zscore_bufs[0].shape[:2] != (h, w):
			self._zscore_bufs = (
				np.empty((h, w, 3), dtype=np.float64),  # hsv
				np.empty((h, w), dtype=np.float64),  # combined
				np.zeros((h, w), dtype=np.uint8),  # mask
			)
		return self._zscore_bufs

	def detect_zscore(self, frame, rows, cols, zscore_threshold=1.5, weights=(0.3, 0.1, 0.6)):
		hsv_buf, combined_buf, mask_buf = self._get_zscore_bufs(frame.shape[0], frame.shape[1])

		hsv_u8 = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
		np.copyto(hsv_buf, hsv_u8, casting='unsafe')
		hsv = hsv_buf

		n_rows, n_cols = rows, cols
		tile_h = hsv.shape[0] // n_rows
		tile_w = hsv.shape[1] // n_cols

		_, global_stdev = cv.meanStdDev(hsv)
		global_stdev = np.maximum(global_stdev.ravel(), 1e-6)

		row_bounds = [(i * tile_h, (i + 1) * tile_h) for i in range(n_rows)]
		col_bounds = [(j * tile_w, (j + 1) * tile_w) for j in range(n_cols)]

		for y0, y1 in row_bounds:
			for x0, x1 in col_bounds:
				tile = hsv[y0:y1, x0:x1]

				mean, local_stdev = cv.meanStdDev(tile)
				stdev = np.maximum(local_stdev.ravel(), global_stdev)

				tile -= mean.ravel() #cv.subtract(tile, mean.ravel(), dst=tile)
				tile /= stdev #cv.divide(tile, stdev, dst=tile)

		hsv[:, :, :2] = np.abs(hsv[:, :, :2])
		hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, None)

		np.multiply(hsv[:, :, 0], weights[0], out=combined_buf)
		combined_buf += hsv[:, :, 1] * weights[1]
		combined_buf += hsv[:, :, 2] * weights[2]

		mask_buf[:] = 0
		mask_buf[combined_buf > zscore_threshold] = 255
		return mask_buf

	def detect(self, frame: np.ndarray, frame_idx: int):
		"""
		Performs object detection on a given frame using SORT tracking.

		Args:
			frame (np.ndarray): The input video frame.
			frame_idx (int): The current frame index.

		Returns:
			A tuple containing the annotated frame and the foreground mask.
		"""
		frame_blurred = cv.GaussianBlur(frame, self.kernel, 0)
		frame_gray_mask = self.detect_zscore(frame_blurred, 6, 5, 3)
		#cv.imshow('graymask', frame_gray_mask)
		fgmask = self.detect_zscore(frame_blurred, 5, 4)
		morph_kernel = cv.getStructuringElement(cv.MORPH_RECT, (11, 11))
		morph_kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
		fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, morph_kernel)  # fills holes
		fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, morph_kernel2)  # removes small noise
		frame_copy = frame.copy()
		fgmask_copy = fgmask.copy()

		contours, h = cv.findContours(
			fgmask_copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
		)

		detections = []

		DIL = 5
		for contour in sorted(contours, key=lambda x: -cv.contourArea(x)):
			x, y, width, height = cv.boundingRect(contour)
			x1, y1, x2, y2 = x, y, x + width, y + height
			area = cv.contourArea(contour)
			if area < 20 or area > 2000:
				continue

			dil_y1 = int(max(y - height // 2 * DIL, 0))
			dil_y2 = int(min(y + height + height // 2 * DIL, frame.shape[0]))
			dil_x1 = int(max(x - width // 2 * DIL, 0))
			dil_x2 = int(min(x + width + width // 2 * DIL, frame.shape[1]))
			dil = frame_blurred[dil_y1: dil_y2, dil_x1: dil_x2]

			filtered = self.detect_zscore(dil, 1, 1, zscore_threshold=2, weights=(0.4, 0.3, 0.3))
			#cv.imshow('filt', filtered)
			#cv.imshow('dil', dil)
			#cv.imshow('orig', frame_blurred[y:y + height, x:x + width])
			filtered = filtered[int(min(height // 2 * DIL, y)): int(min(height // 2 * DIL, y)) + height, int(
				min(width // 2 * DIL, x)): int(min(width // 2 * DIL, x)) + width]
			#cv.imshow('filt_crop', filtered)

			cv.rectangle(frame_copy,
						 tuple(reversed((max(int(y - height // 2 * DIL), 0), max(int(x - width // 2 * DIL), 0)))),
						 tuple(reversed((min(y + height + int(height // 2 * DIL), frame.shape[0]),
										 min(x + width + int(width // 2 * DIL), frame.shape[1])))), (255, 0, 0), 2)

			if np.count_nonzero(filtered) / (np.multiply.reduce(filtered.shape)) < 0.15:
				continue

			fgmask_copy[y: y + height, x: x + width] = cv.bitwise_and(
				fgmask[y: y + height, x: x + width], filtered
			)

			cv.putText(
				frame_copy,
				f"ID: id",
				(x1, y1 - 10),
				cv.FONT_HERSHEY_SIMPLEX,
				0.5,
				(0, 255, 0),
				2,
			)

			detections.append(
				{"bbox": [x1, y1, x2, y2], "conf": 1.0, "frame_idx": frame_idx,
				 'chip': [dil_x1, dil_y1, dil_x2, dil_y2]}
			)

		return detections

	def set_attr(self, name: str, val: float):
		"""
		Sets an attribute of the Detection class.

		Args:
			name (str): The name of the attribute to set.
			val (float): The value to set.

		Returns:
			The new value of the attribute if needed.
		"""
		val = min(self.max_val, max(0, val))
		setattr(self, name, val)
		return getattr(self, name)
