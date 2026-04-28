import math
import os
from pymavlink.dialects.v20 import ardupilotmega as mavlink
from pymavlink import mavutil

os.environ["MAVLINK20"] = "1"


class MavlinkReconnect:
	def __init__(self, endpoint: str, mav_id: int) -> None:
		self.endpoint = endpoint
		self.mav_id = mav_id

	def try_connection(self) -> mavutil.mavlink:
		try:
			connection = mavutil.mavlink_connection(
				self.endpoint, baud=57600,
				planner_format=False,
				notimestamps=True,
				robust_parsing=True,
				dialect="ardupilotmega",
				source_system=self.mav_id,
				source_component=1,
			)
			print("Connection made.")
			return connection
		except ConnectionError as e:
			print(
				f"Error reconnecting: {e}. Attempting to reconnect again..."
			)
			return self.try_connection()


class Drone:
	mav_id: int = 253

	def __init__(self, endpoint, interval):
		self.endpoint = endpoint
		self.interval = interval
		self.mavlink_reconnect = MavlinkReconnect(self.endpoint, self.mav_id)

		self.connection = self.mavlink_reconnect.try_connection()
		self.sent_position_request = False
		self.lat = 0
		self.lon = 0
		self.alt = 0
		self.yaw = 0
		self.pitch = 0
		self.roll = 0
		self.gimbal_yaw = 0
		self.gimbal_pitch = 0
		self.gimbal_roll = 0

	def send_heartbeat(self):
		try:
			self.connection.mav.heartbeat_send(
				mavlink.MAV_TYPE_GCS,
				mavlink.MAV_AUTOPILOT_INVALID,
				0,
				0,
				mavlink.MAV_STATE_ACTIVE,
			)
		except ConnectionError as e:
			print(f"Error sending heartbeat: {e}. Attempting reconnection...")
			self.connection = self.mavlink_reconnect.try_connection()

	def receive_messages(self):
		while True:
			try:
				msg = self.connection.recv_match()

				if not msg:
					break

				if msg.get_type() == "BAD_DATA":
					print("Received bad data")
					continue
				elif msg.get_type() == "HEARTBEAT":
					print(f"Heartbeat received from {msg.get_srcSystem()}")
					if not self.sent_position_request and msg.get_srcSystem() == 1:
						# request position messages every interval
						self.connection.mav.command_long_send(
								self.connection.target_system,
								self.connection.target_component,
								mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
								0,
								mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
								self.interval,
								0,
								0,
								0,
								0,
								0,
								0,
							)
						self.connection.mav.command_long_send(
								self.connection.target_system,
								self.connection.target_component,
								mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
								0,
								mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR,
								self.interval,
								0,
								0,
								0,
								0,
								0,
								0,
							)
						self.connection.mav.command_long_send(
								self.connection.target_system,
								self.connection.target_component,
								mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
								0,
								mavlink.MAVLINK_MSG_ID_ATTITUDE,
								self.interval,
								0,
								0,
								0,
								0,
								0,
								0,
							)
						self.connection.mav.command_long_send(
								self.connection.target_system,
								self.connection.target_component,
								mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
								0,
								mavlink.MAVLINK_MSG_ID_GIMBAL_DEVICE_ATTITUDE_STATUS,
								self.interval,
								0,
								0,
								0,
								0,
								0,
								0,
							)
						self.sent_position_request = True
				elif msg.get_type() == "COMMAND_ACK":
					# if the command wasn't accepted, retry on next heartbeatm
					if (
							msg.command == mavlink.MAV_CMD_SET_MESSAGE_INTERVAL
							and msg.result != mavlink.MAV_RESULT_ACCEPTED
					):
						self.sent_position_request = False

				elif msg.get_type() == "GLOBAL_POSITION_INT":
					# altitude is provided as MSL, need to convert to WGS84
					# lat and lon are provided in degE7 and alt is provided in mm
					self.lat = msg.lat / 1e7
					self.lon = msg.lon / 1e7
					self.yaw = math.radians(0 if msg.hdg == 2 ** 16 - 1 else msg.hdg / 100)

				elif msg.get_type() == "DISTANCE_SENSOR":
					self.alt = msg.current_distance / 100
				elif msg.get_type() == "ATTITUDE":
					self.roll = msg.roll + math.pi
					self.pitch = msg.pitch + math.pi
				elif msg.get_type() == "GIMBAL_DEVICE_ATTITUDE_STATUS":
					self.gimbal_roll = math.atan2(2 * (msg.q[0] * msg.q[1] + msg.q[2] * msg.q[3]),
													  1 - 2 * (msg.q[1] ** 2 + msg.q[2] ** 2))
					self.gimbal_pitch = math.asin(2 * (msg.q[0] * msg.q[2] - msg.q[3] * msg.q[1]))
					self.gimbal_yaw = math.atan2(2 * (msg.q[0] * msg.q[3] + msg.q[1] * msg.q[2]),
													 1 - 2 * (msg.q[2] ** 2 + msg.q[3] ** 2))
			except ConnectionError as e:
				print(f"Error receiving message: {e}. Attempting reconnection...")
				self.connection = self.mavlink_reconnect.try_connection()
				self.sent_position_request = False

	def getLLA(self):
		return self.lat, self.lon, self.alt

	def getValues(self):
		return 0, 0, self.alt

	def getRollPitchYaw(self):
		return self.roll, self.pitch, self.yaw

	def getGimbalRollPitchYaw(self):
		return self.gimbal_roll, self.gimbal_pitch, self.gimbal_yaw
