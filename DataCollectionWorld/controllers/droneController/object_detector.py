"""Object detection and 3D-to-2D projection module for Webots."""
import functools

import numpy as np
import math
from typing import List, Tuple, Optional, Dict


class ObjectDetector:
    def __init__(self, camera, supervisor):
        """
        Initialize the object detector.
        
        Args:
            camera: Webots camera device
            supervisor: Webots supervisor instance
        """
        self.camera = camera
        self.supervisor = supervisor
        self.cam_width = camera.getWidth()
        self.cam_height = camera.getHeight()
        self.fov = camera.getFov()
        
        # Calculate camera intrinsic matrix
        self.K = self._get_camera_intrinsic()
    
    def _get_camera_intrinsic(self):
        """Calculate camera intrinsic matrix from FOV."""
        cx = (self.cam_width - 1) / 2
        cy = (self.cam_height - 1) / 2
        fx = self.cam_width / (2 * math.tan(self.fov / 2))
        fy = fx
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def get_camera_transform(self, imu, gps, cam_pitch_offset=0.0, cam_yaw_offset=0.0):
        """
        Get the camera transformation matrix (rotation and translation).
        
        Args:
            imu: IMU device
            gps: GPS device
            cam_pitch_offset: Camera pitch offset
            cam_yaw_offset: Camera yaw offset
        
        Returns:
            Tuple of (R, t) where R is rotation matrix and t is translation vector
        """
        # Get drone pose
        pos = np.array(gps.getValues())
        roll, pitch, yaw = imu.getRollPitchYaw()
        
        # Camera gimbal rotation
        R_cam_yaw = np.array([
            [np.cos(cam_yaw_offset), -np.sin(cam_yaw_offset), 0],
            [np.sin(cam_yaw_offset), np.cos(cam_yaw_offset), 0],
            [0, 0, 1]
        ])
        
        R_cam_pitch = np.array([
            [np.cos(cam_pitch_offset), 0, np.sin(cam_pitch_offset)],
            [0, 1, 0],
            [-np.sin(cam_pitch_offset), 0, np.cos(cam_pitch_offset)]
        ])
        
        R_cam = R_cam_yaw @ R_cam_pitch
        
        # Drone body rotation
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Combined rotation
        R = Rz @ Ry @ Rx @ R_cam
        t = pos.reshape(3, 1)
        
        # Transform to camera coordinates
        t = -R.T @ t
        R = R.T
        
        return R, t
    
    def project_3d_to_2d(self, point_3d, R, t):
        """
        Project a 3D point to 2D camera coordinates.
        
        Args:
            point_3d: 3D point as (x, y, z)
            R: Camera rotation matrix
            t: Camera translation vector
        
        Returns:
            Tuple of (u, v, depth) or None if point is behind camera
        """
        point_3d = np.array(point_3d).reshape(3, 1)
        
        # Transform to camera coordinates
        point_cam = R @ point_3d + t
        
        # Check if point is in front of camera
        if point_cam[2, 0] <= 0:
            return None
        
        # Project to image plane
        point_2d_homogeneous = self.K @ point_cam
        u = int(point_2d_homogeneous[0, 0] / point_2d_homogeneous[2, 0])
        v = int(point_2d_homogeneous[1, 0] / point_2d_homogeneous[2, 0])
        depth = point_cam[2, 0]
        
        return u, v, depth
    
    def is_in_view(self, u, v, margin=0):
        """Check if a 2D point is within the camera view."""
        return (margin <= u < self.cam_width - margin and 
                margin <= v < self.cam_height - margin)
    
    def calculate_bbox_from_sphere(self, center_3d, radius, R, t):
        """
        Calculate 2D bounding box for a sphere.
        
        Args:
            center_3d: 3D center position of sphere
            radius: Sphere radius
            R: Camera rotation matrix
            t: Camera translation vector
        
        Returns:
            Tuple of (x, y, width, height) or None if not visible
        """
        # Project sphere center
        projection = self.project_3d_to_2d(center_3d, R, t)
        if projection is None:
            return None
        
        center_u, center_v, depth = projection
        
        # Calculate apparent radius in pixels based on depth
        # Using pinhole camera model: pixel_radius = (focal_length * real_radius) / depth
        focal_length = self.K[0, 0]
        pixel_radius = (focal_length * radius) / depth
        pixel_radius = int(pixel_radius)
        
        # Calculate bounding box
        x = int(center_u - pixel_radius)
        y = int(center_v - pixel_radius)
        width = 2 * pixel_radius
        height = 2 * pixel_radius
        
        # Check if at least part of the sphere is visible
        if (x + width < 0 or x >= self.cam_width or 
            y + height < 0 or y >= self.cam_height):
            return None
        
        # Clamp to image boundaries
        x = max(0, min(x, self.cam_width - 1))
        y = max(0, min(y, self.cam_height - 1))
        
        # Adjust width and height if clamped
        width = min(width, self.cam_width - x)
        height = min(height, self.cam_height - y)
        
        return (x, y, width, height)
    
    def find_spheres(self):
        """
        Find all sphere objects in the Webots world.
        
        Returns:
            List of dictionaries with sphere information
        """
        spheres = []
        
        # Check if supervisor mode is available
        try:
            root = self.supervisor.getRoot()
        except AttributeError:
            print("ERROR: This controller requires Supervisor mode.")
            print("Please set 'supervisor TRUE' in the robot definition in your .wbt file")
            return []
        
        if root is None:
            print("ERROR: Could not get root node. Supervisor mode may not be enabled.")
            return []
        
        children_field = root.getField("children")
        if children_field is None:
            print("ERROR: Could not get children field from root node.")
            return []
        
        n = children_field.getCount()
        
        for i in range(n):
            node = children_field.getMFNode(i)
            if node is None:
                continue
            
            type_name = node.getTypeName()
            
            # Check if this is a Solid node (our spawned spheres)
            if type_name == "Solid":
                # Try to get translation field
                translation_field = node.getField("translation")
                if translation_field is None:
                    continue
                
                position = translation_field.getSFVec3f()
                
                # Check if it has a Shape child with Sphere geometry
                children = node.getField("children")
                if children is None:
                    continue
                
                is_sphere = False
                color = None
                radius = 0.5  # Default radius
                
                for j in range(children.getCount()):
                    child = children.getMFNode(j)
                    if child and child.getTypeName() == "Shape":
                        # Check geometry
                        geometry_field = child.getField("geometry")
                        if geometry_field:
                            geometry = geometry_field.getSFNode()
                            if geometry and geometry.getTypeName() == "Sphere":
                                is_sphere = True
                                # Get radius
                                radius_field = geometry.getField("radius")
                                if radius_field:
                                    radius = radius_field.getSFFloat()
                        
                        # Check color
                        appearance_field = child.getField("appearance")
                        if appearance_field:
                            appearance = appearance_field.getSFNode()
                            if appearance:
                                base_color_field = appearance.getField("baseColor")
                                if base_color_field:
                                    color = base_color_field.getSFColor()
                
                if is_sphere and color is not None:
                    # Determine color type (check if it's red)
                    r, g, b = color
                    color_type = "unknown"
                    if r > 0.8 and g < 0.3 and b < 0.3:
                        color_type = "red"
                    elif g > 0.8 and r < 0.3 and b < 0.3:
                        color_type = "green"
                    elif b > 0.8 and r < 0.3 and g < 0.3:
                        color_type = "blue"
                    
                    spheres.append({
                        'node': node,
                        'position': position,
                        'radius': radius,
                        'color': color,
                        'color_type': color_type,
                        'name': f"sphere_{i}"
                    })
        
        return spheres
    
    def detect_visible_spheres(self, imu, gps, cam_pitch_offset=0.0, cam_yaw_offset=0.0,
                               filter_color=None):
        """
        Detect all visible spheres in the current camera view.
        
        Args:
            imu: IMU device
            gps: GPS device
            cam_pitch_offset: Camera pitch offset
            cam_yaw_offset: Camera yaw offset
            filter_color: Only detect spheres of this color (e.g., "red")
        
        Returns:
            List of detection dictionaries
        """
        # Get camera transform
        R, t = self.get_camera_transform(imu, gps, cam_pitch_offset, cam_yaw_offset)
        
        # Find all spheres
        spheres = self.find_spheres()
        
        detections = []
        for sphere in spheres:
            # Filter by color if specified
            if filter_color and sphere['color_type'] != filter_color:
                continue
            
            # Calculate bounding box
            bbox = self.calculate_bbox_from_sphere(sphere['position'], sphere['radius'], R, t)
            
            if bbox is not None:
                detections.append({
                    'name': sphere['name'],
                    'type': f"{sphere['color_type']}_sphere",
                    'bbox': bbox,
                    'position_3d': tuple(sphere['position']),
                    'color': sphere['color'],
                    'radius': sphere['radius']
                })
        
        return detections
	@functools.lru_cache

