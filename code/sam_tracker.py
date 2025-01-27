import os
import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sam2.build_sam import build_sam2_video_predictor

@dataclass
class VideoInfo:
    """
    Data class storing video metadata.
    
    Attributes:
        width: Video frame width in pixels
        height: Video frame height in pixels
        frame_count: Total number of frames in video
        fps: Frames per second (default: 30.0)
    """
    width: int
    height: int
    frame_count: int
    fps: float = 30.0

class HandTracker:
    """
    Tracks hands through video using SAM2 (Segment Anything Model 2.0):
        1. Frame extraction
        2. Tracking initialization
        3. Mask propagation
        4. Visualization
    """
    def __init__(self, sam2_checkpoint: str, model_cfg: str, device: str = None):
        """
        Initialize the HandTracker with SAM2 model.
        
        Args:
            sam2_checkpoint: Path to pretrained SAM2 weights
            model_cfg: Path to model configuration file
            device: Computing device ('cuda' or 'cpu'). If None, uses CUDA if available
        """
        self.device = self._initialize_device(device)
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)

    def _initialize_device(self, device: Optional[str]) -> torch.device:
        """
        Initialize the processing device (CPU/GPU).
        
        Args:
            device: Optional device specification
            
        Returns:
            torch.device: CUDA device if available and not explicitly set to CPU
        """
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _extract_frames(self, input_path: str, temp_dir: str) -> VideoInfo:
        """
        Extract individual frames from video file and save to disk temporarily.
        
        Args:
            input_path: Path to input video file
            temp_dir: Directory to store extracted frames
            
        Returns:
            VideoInfo: Object containing video metadata

        """
        os.makedirs(temp_dir, exist_ok=True)
        cap = cv2.VideoCapture(input_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Frames are saved as JPEG files with 5-digit zero-padded indices
            cv2.imwrite(f"{temp_dir}/{frame_count:05d}.jpg", frame)
            frame_count += 1
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        return VideoInfo(width, height, frame_count, fps)
    
    def _initialize_tracking(self, temp_dir: str, first_frame_boxes: List[List[float]]) -> Tuple[dict, dict]:
        """
        Initialize SAM2 tracking state with initial hand bounding boxes.
        
        Args:
            temp_dir: Directory containing extracted frames
            first_frame_boxes: List of hand bounding boxes [x1, y1, x2, y2] from first frame
            
        Returns:
            Tuple containing:
                - inference_state: SAM2 internal tracking state
                - video_segments: Empty dict to store tracking results
        """
        inference_state = self.predictor.init_state(video_path=temp_dir)
        video_segments = {}

        for i, box in enumerate(first_frame_boxes):
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=i,
                box=box
            )

        return inference_state, video_segments
    
    def _propagate_masks(self, inference_state: dict) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Propagate hand segmentation masks through video frames.
        
        Args:
            inference_state: SAM2 tracking state
            
        Returns:
            Dict mapping frame indices to masks for each tracked hand:
            {frame_idx: {hand_id: binary_mask}}
            
        Note:
            Masks are binary numpy arrays where True indicates hand pixels
        """
        video_segments = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
        return video_segments
    
    def _create_output_video(self, temp_dir: str, output_path: str, video_info: VideoInfo, 
                             video_segments: Dict[int, Dict[int, np.ndarray]]) -> None:
        """
        Create visualization video with highlighted hand regions.
        
        Args:
            temp_dir: Directory containing source frames
            output_path: Path to save output video
            video_info: Video metadata
            video_segments: Dictionary of hand masks per frame
            
        Note:
            Hands are highlighted in blue with 30% opacity overlay
        """
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                                 video_info.fps, (video_info.width, video_info.height))

        for frame_idx in range(video_info.frame_count):
            frame = cv2.imread(f"{temp_dir}/{frame_idx:05d}.jpg")
            if frame_idx in video_segments:
                for mask in video_segments[frame_idx].values():
                    frame[mask] = frame[mask] * 0.7 + np.array([255, 0, 0]) * 0.3
            writer.write(frame)
            
        writer.release()

    def track_video(self, input_path: str, output_path: str, first_frame_boxes: list):
        """
        Main method to track hands through video using SAM2.
        
        Args:
            input_path: Path to input video file
            output_path: Path to save processed video
            first_frame_boxes: List of initial hand bounding boxes [x1, y1, x2, y2]
        """
        
        temp_dir = "temp_frames"
        
        try:
            # Extract frames and get video info
            video_info = self._extract_frames(input_path, temp_dir)
            
            # Initialize tracking with first frame boxes
            inference_state, video_segments = self._initialize_tracking(temp_dir, first_frame_boxes)
            
            # Propagate masks through video
            video_segments = self._propagate_masks(inference_state)
            
            # Create output video with visualizations
            self._create_output_video(temp_dir, output_path, video_info, video_segments)
            
        finally:
            # Cleanup temporary files
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)