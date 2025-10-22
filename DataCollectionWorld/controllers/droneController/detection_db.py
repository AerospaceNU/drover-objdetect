"""Database module for object detection labels."""

import sqlite3
import os
from datetime import datetime
from typing import List, Tuple, Optional


class DetectionDatabase:
    def __init__(self, db_path: str = "detections.db"):
        """Initialize the database connection and create tables if needed."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        """Create the necessary tables for storing detection data."""
        # Main detections table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_idx INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                object_name TEXT NOT NULL,
                object_type TEXT NOT NULL,
                bbox_x INTEGER NOT NULL,
                bbox_y INTEGER NOT NULL,
                bbox_width INTEGER NOT NULL,
                bbox_height INTEGER NOT NULL,
                pos_x REAL NOT NULL,
                pos_y REAL NOT NULL,
                pos_z REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Index for faster frame lookups
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_frame_idx 
            ON detections(frame_idx)
        """)
        
        # Session metadata table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_frames INTEGER DEFAULT 0,
                total_detections INTEGER DEFAULT 0,
                raw_video_path TEXT,
                labeled_video_path TEXT
            )
        """)
        
        self.conn.commit()
    
    def create_session(self, session_name: str) -> int:
        """Create a new detection session and return its ID."""
        now = datetime.now().isoformat()
        self.cursor.execute("""
            INSERT INTO sessions (session_name, start_time)
            VALUES (?, ?)
        """, (session_name, now))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def update_session(self, session_id: int, total_frames: int, total_detections: int,
                      raw_video_path: str = None, labeled_video_path: str = None):
        """Update session metadata when finished."""
        now = datetime.now().isoformat()
        self.cursor.execute("""
            UPDATE sessions 
            SET end_time = ?,
                total_frames = ?,
                total_detections = ?,
                raw_video_path = ?,
                labeled_video_path = ?
            WHERE id = ?
        """, (now, total_frames, total_detections, raw_video_path, labeled_video_path, session_id))
        self.conn.commit()
    
    def log_detection(self, frame_idx: int, timestamp: float, object_name: str,
                     object_type: str, bbox: Tuple[int, int, int, int],
                     position_3d: Tuple[float, float, float]):
        """
        Log a single detection to the database.
        
        Args:
            frame_idx: Frame index in the video
            timestamp: Simulation timestamp
            object_name: Name/ID of the detected object
            object_type: Type of object (e.g., "red_sphere")
            bbox: Bounding box as (x, y, width, height)
            position_3d: 3D position as (x, y, z)
        """
        now = datetime.now().isoformat()
        bbox_x, bbox_y, bbox_width, bbox_height = bbox
        pos_x, pos_y, pos_z = position_3d
        
        self.cursor.execute("""
            INSERT INTO detections 
            (frame_idx, timestamp, object_name, object_type, 
             bbox_x, bbox_y, bbox_width, bbox_height,
             pos_x, pos_y, pos_z, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (frame_idx, timestamp, object_name, object_type,
              bbox_x, bbox_y, bbox_width, bbox_height,
              pos_x, pos_y, pos_z, now))
        
        self.conn.commit()
    
    def log_detections_batch(self, detections: List[dict]):
        """Log multiple detections at once for better performance."""
        now = datetime.now().isoformat()
        data = [
            (d['frame_idx'], d['timestamp'], d['object_name'], d['object_type'],
             d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3],
             d['position_3d'][0], d['position_3d'][1], d['position_3d'][2], now)
            for d in detections
        ]
        
        self.cursor.executemany("""
            INSERT INTO detections 
            (frame_idx, timestamp, object_name, object_type,
             bbox_x, bbox_y, bbox_width, bbox_height,
             pos_x, pos_y, pos_z, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        
        self.conn.commit()
    
    def get_detections_for_frame(self, frame_idx: int) -> List[dict]:
        """Retrieve all detections for a specific frame."""
        self.cursor.execute("""
            SELECT * FROM detections WHERE frame_idx = ?
        """, (frame_idx,))
        
        rows = self.cursor.fetchall()
        detections = []
        for row in rows:
            detections.append({
                'id': row[0],
                'frame_idx': row[1],
                'timestamp': row[2],
                'object_name': row[3],
                'object_type': row[4],
                'bbox': (row[5], row[6], row[7], row[8]),
                'position_3d': (row[9], row[10], row[11]),
                'created_at': row[12]
            })
        return detections
    
    def get_statistics(self) -> dict:
        """Get overall detection statistics."""
        self.cursor.execute("SELECT COUNT(*) FROM detections")
        total_detections = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(DISTINCT frame_idx) FROM detections")
        frames_with_detections = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(DISTINCT object_name) FROM detections")
        unique_objects = self.cursor.fetchone()[0]
        
        return {
            'total_detections': total_detections,
            'frames_with_detections': frames_with_detections,
            'unique_objects': unique_objects
        }
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

