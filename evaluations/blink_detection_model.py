import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Dict, Optional, Sequence
import json

class BlinkDetectionModel:
    """
    Modular blink detection model using MediaPipe Face Mesh and EAR algorithm.
    Designed for both real-time and batch evaluation.
    """
    
    def __init__(self, 
                 ear_threshold: float = 0.21,
                 consecutive_frames: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the blink detection model.
        
        Args:
            ear_threshold: EAR threshold for blink detection
            consecutive_frames: Number of consecutive frames below threshold to confirm blink
            min_detection_confidence: MediaPipe face detection confidence
            min_tracking_confidence: MediaPipe face tracking confidence
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Eye landmark indices for MediaPipe Face Mesh
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        
    def _euclidean_dist(self, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return float(np.linalg.norm(np.array(pt1) - np.array(pt2)))
    
    def _eye_aspect_ratio(self, eye_landmarks: Sequence[Tuple[float, float]]) -> float:
        """
        Compute the Eye Aspect Ratio (EAR).
        
        Args:
            eye_landmarks: Sequence of 6 eye landmark points
            
        Returns:
            EAR value
        """
        # Compute the vertical distances
        A = self._euclidean_dist(eye_landmarks[1], eye_landmarks[5])
        B = self._euclidean_dist(eye_landmarks[2], eye_landmarks[4])
        # Compute the horizontal distance
        C = self._euclidean_dist(eye_landmarks[0], eye_landmarks[3])
        
        # EAR = (A + B) / (2.0 * C)
        ear = (A + B) / (2.0 * C)
        return ear
    
    def _initialize_face_mesh(self):
        """Initialize MediaPipe Face Mesh if not already done."""
        if self.face_mesh is None:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame and return blink detection results.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary containing:
            - ear: Eye aspect ratio
            - blink_detected: Boolean indicating if blink was detected
            - face_detected: Boolean indicating if face was detected
            - left_ear: Left eye EAR
            - right_ear: Right eye EAR
        """
        self._initialize_face_mesh()
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return {
                'ear': None,
                'blink_detected': False,
                'face_detected': False,
                'left_ear': None,
                'right_ear': None
            }
        
        h, w, _ = frame.shape
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract eye landmarks
        left_eye = [(int(face_landmarks.landmark[i].x * w), 
                     int(face_landmarks.landmark[i].y * h)) for i in self.LEFT_EYE]
        right_eye = [(int(face_landmarks.landmark[i].x * w), 
                      int(face_landmarks.landmark[i].y * h)) for i in self.RIGHT_EYE]
        
        # Calculate EAR for both eyes
        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        return {
            'ear': ear,
            'blink_detected': ear < self.ear_threshold,
            'face_detected': True,
            'left_ear': left_ear,
            'right_ear': right_ear
        }
    
    def process_video(self, video_path: str, output_frames: bool = False) -> Dict:
        """
        Process an entire video and return blink detection results.
        
        Args:
            video_path: Path to the video file
            output_frames: Whether to return per-frame results
            
        Returns:
            Dictionary containing:
            - total_frames: Total number of frames processed
            - frames_with_face: Number of frames where face was detected
            - frames_with_blink: Number of frames where blink was detected
            - ear_values: List of EAR values (if output_frames=True)
            - blink_frames: List of frame indices where blinks occurred
            - average_ear: Average EAR across all frames with face detected
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = 0
        frames_with_face = 0
        frames_with_blink = 0
        ear_values = []
        blink_frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = self.process_frame(frame)
                total_frames += 1
                
                if result['face_detected']:
                    frames_with_face += 1
                    ear_values.append(result['ear'])
                    
                    if result['blink_detected']:
                        frames_with_blink += 1
                        blink_frames.append(total_frames - 1)
                
        finally:
            cap.release()
        
        # Calculate average EAR
        average_ear = np.mean(ear_values) if ear_values else 0.0
        
        return {
            'total_frames': total_frames,
            'frames_with_face': frames_with_face,
            'frames_with_blink': frames_with_blink,
            'ear_values': ear_values if output_frames else None,
            'blink_frames': blink_frames,
            'average_ear': average_ear,
            'face_detection_rate': frames_with_face / total_frames if total_frames > 0 else 0.0,
            'blink_rate': frames_with_blink / frames_with_face if frames_with_face > 0 else 0.0
        }
    
    def cleanup(self):
        """Clean up MediaPipe resources."""
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None 