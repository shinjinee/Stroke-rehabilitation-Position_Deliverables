import mediapipe as mp
import numpy as np
import cv2
from typing import List

class HandDetector:
    """
    A class for detecting hands in images/video frames using MediaPipe Hands.
    Returns bounding boxes compatible with SAM2 format for further processing.
    """
    def __init__(self, min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7,
    ):
        """
        Initialize the HandDetector with MediaPipe configuration.
        
        Args:
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand detection to be considered successful.
            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the hand landmarks to be considered tracked successfully.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame: np.ndarray) -> List[List[float]]:
        """
        Detect hands in a frame and return their bounding boxes.
        
        Args:
            frame: Input image/video frame in BGR format (OpenCV default)
        
        Returns:
            List of bounding boxes in format [x1, y1, x2, y2] where:
                x1, y1: Top-left corner coordinates
                x2, y2: Bottom-right corner coordinates
            Returns empty list if no hands detected.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        boxes = []
        
        # If hands are detected, compute their bounding boxes
        if results.multi_hand_landmarks:
            height, width = frame.shape[:2]

            for hand_landmarks in results.multi_hand_landmarks:
                # Convert normalized landmarks to pixel coordinates
                keypoints = np.array(
                    [[lm.x * width, lm.y * height] for lm in hand_landmarks.landmark]
                )
                
                # Compute bounding box coordinates with padding & ensure coordinates don't exceed image boundaries
                padding = 20
                x1 = max(0, float(np.min(keypoints[:, 0])) - padding)
                y1 = max(0, float(np.min(keypoints[:, 1])) - padding)
                x2 = min(width, float(np.max(keypoints[:, 0])) + padding)
                y2 = min(height, float(np.max(keypoints[:, 1])) + padding)
                
                boxes.append([x1, y1, x2, y2])
        
        return boxes

    def __del__(self):
        self.hands.close()