import math

import cv2 as cv

from locator.locator import Locator
from droverDetection.detection import Detection
from locator.camera import Camera
from locator.drone import Drone
import dataclasses
import numpy as np
import struct
import websockets.sync.server
import threading
import sys


@dataclasses.dataclass
class TrackedObject:
	lat: float
	lon: float
	bbox: tuple#[int, int, int, int]
	chip: np.ndarray
	seen_count: int = 0


class Tracker:
	def __init__(self):
		self.camera = Camera(math.radians(160))
		self.camera.getCameraIntrinsic()
		print(self.camera.newcameramtx, self.camera.getEstimatedCameraIntrinsic())
		self.drone = Drone('/dev/ttyTHS1', int(20000))
		self.locator = Locator(self.drone, self.camera)
		self.detection = Detection()
		self.tracked_objects = []
		self.clients = []

	@staticmethod
	def pack_tracked_obj(i:int, obj: TrackedObject):
		ret, chip = cv.imencode('.png', obj.chip)
		if ret:
			return struct.pack('BffHHHHI', i, obj.lat, obj.lon, *obj.bbox, len(chip)) + chip.tobytes()
		return None

	def get_closest_object(self, ref_obj: TrackedObject):
		for i, obj in enumerate(self.tracked_objects):
			a = math.sin((obj.lat - ref_obj.lat) / 2) ** 2 + math.cos(ref_obj.lat) * math.cos(obj.lat) * math.sin(
				(obj.lon - ref_obj.lon) / 2) ** 2
			c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
			d = 6371000 * c
			if d < 2:
				return i, obj
		return None, None

	def run(self):
		while True:
			self.drone.send_heartbeat()
			self.drone.receive_messages()
			ret, frame = self.camera.read()
			if not ret:
				break
			detections = self.detection.detect(frame, 0)
			updated_obj_idxs = []
			for detection in detections:
				bbox = detection['bbox']
				x = (bbox[2] - bbox[0]) / 2
				y = (bbox[3] - bbox[1]) / 2
				lat, lon, _ = self.locator.point_cam_to_LLA(x, y)
				chip = detection['chip']
				obj = TrackedObject(lat, lon, bbox, frame[chip[1]:chip[3], chip[0]:chip[2]])
				obj_idx, seen_obj = self.get_closest_object(obj)
				if seen_obj:
					seen_obj.lat = (seen_obj.lat * seen_obj.seen_count + obj.lat) / (seen_obj.seen_count + 1)
					seen_obj.lon = (seen_obj.lon * seen_obj.seen_count + obj.lon) / (seen_obj.seen_count + 1)
					seen_obj.bbox = (min(seen_obj.bbox[0], obj.bbox[0]), min(seen_obj.bbox[1], obj.bbox[1]),
									 max(seen_obj.bbox[2], obj.bbox[2]), max(seen_obj.bbox[3], obj.bbox[3]))
					seen_obj.seen_count += 1
					seen_obj.chip = obj.chip
				else:
					self.tracked_objects.append(obj)
					obj_idx = len(self.tracked_objects) - 1
				updated_obj_idxs.append(obj_idx)
			message = b''
			message += struct.pack('B', len(updated_obj_idxs))
			for obj_idx in updated_obj_idxs:
				message += self.pack_tracked_obj(obj_idx, self.tracked_objects[obj_idx])
				obj = self.tracked_objects[obj_idx]
				print(obj.lat, obj.lon, obj.seen_count)
			for client in self.clients:
				client.send(message)

tracker = None

def handler(websocket):
	tracker.clients.append(websocket)

def run_server():
	with websockets.sync.server.serve(handler, '0.0.0.0', 8765) as server:
		server.serve_forever()

if __name__ == '__main__':
	if len(sys.argv) == 1:
		tracker = Tracker()
		t = threading.Thread(target=run_server)
		t.start()
		tracker.run()
		t.join()
	else:
		camera = Camera(math.radians(160))
		camera.calculateCameraIntrinsic()
