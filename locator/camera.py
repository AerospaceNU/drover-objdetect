import cv2 as cv
import math
import numpy as np


class Camera:
	def __init__(self, fov):
		self.camera = cv.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12 ! nvvidconv ! video/x-raw ! videoconvert ! video/x-raw,format=BGR ! appsink', cv.CAP_GSTREAMER)
		print(self.getWidth(), self.getHeight(), self.camera.read()[0])
		self.hfov = 2 * math.atan(
			math.tan(fov / 2) * (self.getWidth() / math.sqrt(self.getWidth() ** 2 + self.getHeight() ** 2)))
		self.vfov = 2 * math.atan(
			math.tan(fov / 2) * (self.getHeight() / math.sqrt(self.getWidth() ** 2 + self.getHeight() ** 2)))
		self.mtx = None
		self.dist = None
		self.newcameramtx = None
		self.roi = None

	def getWidth(self):
		return int(self.camera.get(cv.CAP_PROP_FRAME_WIDTH))

	def getHeight(self):
		return int(self.camera.get(cv.CAP_PROP_FRAME_HEIGHT))

	def getFOV(self):
		return self.hfov

	def getFPS(self):
		return self.camera.get(cv.CAP_PROP_FPS)

	def nativeWidth(self):
		return 4056
	
	def nativeHeight(self):
		return 3040

	def getEstimatedCameraIntrinsic(self):
		cx = (self.nativeWidth() - 1) / 2
		cy = (self.nativeHeight() - 1) / 2
		fx = self.nativeWidth() / (2 * math.tan(self.hfov / 2))
		fy = self.nativeHeight() / (2 * math.tan(self.vfov / 2))
		return np.array([
			[fx, 0, cx],
			[0, fy, cy],
			[0, 0, 1]
		], dtype=np.float32)

	def calculateCameraIntrinsic(self):
		# termination criteria
		criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		objp = np.zeros((6 * 7, 3), np.float32)
		objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

		# Arrays to store object points and image points from all the images.
		objpoints = []  # 3d point in real world space
		imgpoints = []  # 2d points in image plane.

		gray = None
		while True:
			ret, img = self.camera.read()
			if not ret:
				break
			gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
			print('got frame')

			# Find the chess board corners
			ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

			# If found, add object points, image points (after refining them)
			if ret:
				print('got board')
				objpoints.append(objp)

				corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
				imgpoints.append(corners2)

				# Draw and display the corners
				cv.drawChessboardCorners(img, (7, 6), corners2, ret)
				#cv.imshow('img', img)
				#cv.waitKey(500)
				break
		ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
		newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (self.getWidth(), self.getHeight()), 1,
														 (self.getWidth(), self.getHeight()))
		np.save('k.npy', mtx)
		np.save('dist.npy', dist)
		np.save('opt-k.npy', newcameramtx)
		np.save('roi.npy', roi)
		self.mtx = mtx
		self.dist = dist
		self.newcameramtx = newcameramtx
		self.roi = roi

		#cv.destroyAllWindows()

	def getCameraIntrinsic(self):
		self.mtx = np.load('k.npy')
		self.dist = np.load('dist.npy')
		self.newcameramtx = np.load('opt-k.npy')
		self.roi = np.load('roi.npy')

	def read(self):
		ret, frame = self.camera.read()
		if ret:
			frame = cv.undistort(frame, self.mtx, self.dist, None, self.newcameramtx)
			x, y, w, h = self.roi
			return ret, frame[y:y + h, x:x + w]
		return ret, frame
