import cv2
from hand_detector import HandDetector
from sam_tracker import HandTracker
import argparse

def start_detection(input_path: str, output_path: str, 
                  sam_checkpoint: str, model_cfg: str,) -> None:
    """
    Process video file to detect and track hands using a two-stage approach:
    1. Initial hand detection using MediaPipe
    2. Subsequent tracking using SAM 2.1
    
    Args:
        input_path: Path to the input video file
        output_path: Path where the processed video will be saved
        sam_checkpoint: Path to the pretrained SAM 2.1 model weights
        model_cfg: Path to the SAM 2.1 model configuration file
    
    Raises:
        ValueError: If the input video file cannot be opened
    """
    # Initialize MediaPipe hand detector and SAM 2.1 tracker
    hand_detector = HandDetector()
    hand_tracker = HandTracker(sam_checkpoint, model_cfg)

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Process frames until first successful hand detection
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detector only on first frame to get initial hand positions
        detections = hand_detector.detect(frame)
        if detections:
            # Initialize tracking with first frame detections
            hand_tracker.track_video(input_path, output_path, detections)
            break

    # Cleanup resources
    cap.release()

if __name__ == "__main__":
    # Configure command-line argument parser to run from terminal with custom parameters
    parser = argparse.ArgumentParser(
        description="Hand tracking with MediaPipe and SAM 2.1"
    )
    parser.add_argument("input_path", type=str, help="Path to input video")
    parser.add_argument("output_path", type=str, help="Path to output video")
    parser.add_argument("sam_checkpoint", type=str, help="Path to SAM 2.1 checkpoint")
    parser.add_argument("model_cfg", type=str, help="Path to SAM 2.1 configuration")
    args = parser.parse_args()
    
    # Initialize processing pipeline with parsed arguments
    start_detection(args.input_path, args.output_path, 
                    args.sam_checkpoint, args.model_cfg,
)
