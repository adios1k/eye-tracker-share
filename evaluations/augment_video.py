#!/usr/bin/env python3
"""
Data Augmentation Script for Eye Blink Detection Videos

This script creates various augmented versions of the input video to test
the robustness of the blink detection model under different conditions.

Usage:
    python augment_video.py --input video.mp4 --output_dir ./augmented_videos
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import json
from typing import Tuple, Dict, Any
import logging
from numpy.typing import NDArray

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoAugmenter:
    """Class to handle video augmentation for blink detection testing."""
    
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def adjust_brightness(self, frame: Any, factor: float) -> Any:
        """Adjust brightness of the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, frame: Any, factor: float) -> Any:
        """Adjust contrast of the frame."""
        return np.clip(frame * factor, 0, 255).astype(np.uint8)
    
    def add_noise(self, frame: Any, intensity: float = 0.1) -> Any:
        """Add Gaussian noise to the frame."""
        noise = np.random.normal(0, intensity * 255, frame.shape).astype(np.uint8)
        return np.clip(frame + noise, 0, 255).astype(np.uint8)
    
    def apply_blur(self, frame: Any, kernel_size: int = 5) -> Any:
        """Apply Gaussian blur to the frame."""
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    def adjust_saturation(self, frame: Any, factor: float) -> Any:
        """Adjust saturation of the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def add_compression_artifacts(self, frame: np.ndarray, quality: int = 50) -> np.ndarray:
        """Simulate compression artifacts by encoding and decoding."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', frame, encode_param)
        return cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    def resize_frame(self, frame: np.ndarray, scale_factor: float) -> np.ndarray:
        """Resize frame by scale factor."""
        height, width = frame.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return cv2.resize(frame, (new_width, new_height))
    
    def create_augmented_video(self, augmentation_name: str, 
                              augmentation_func, *args, **kwargs) -> str:
        """Create an augmented video using the specified function."""
        cap = cv2.VideoCapture(self.input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output filename
        input_name = Path(self.input_path).stem
        output_path = self.output_dir / f"{input_name}_{augmentation_name}.mp4"
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Apply augmentation
            augmented_frame = augmentation_func(frame, *args, **kwargs)
            out.write(augmented_frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames for {augmentation_name}")
        
        cap.release()
        out.release()
        
        logger.info(f"Created augmented video: {output_path}")
        return str(output_path)
    
    def generate_all_augmentations(self) -> Dict[str, str]:
        """Generate all augmented videos and return their paths."""
        augmentations = {
            "bright_dark": lambda frame: self.adjust_brightness(frame, 0.3),
            "bright_light": lambda frame: self.adjust_brightness(frame, 1.8),
            "low_contrast": lambda frame: self.adjust_contrast(frame, 0.5),
            "high_contrast": lambda frame: self.adjust_contrast(frame, 1.5),
            "noise_light": lambda frame: self.add_noise(frame, 0.05),
            "noise_heavy": lambda frame: self.add_noise(frame, 0.2),
            "blur_light": lambda frame: self.apply_blur(frame, 3),
            "blur_heavy": lambda frame: self.apply_blur(frame, 9),
            "low_saturation": lambda frame: self.adjust_saturation(frame, 0.3),
            "high_saturation": lambda frame: self.adjust_saturation(frame, 1.8),
            "compression_low": lambda frame: self.add_compression_artifacts(frame, 30),
            "compression_medium": lambda frame: self.add_compression_artifacts(frame, 60),
            "scale_small": lambda frame: self.resize_frame(frame, 0.5),
            "scale_large": lambda frame: self.resize_frame(frame, 1.5)
        }
        
        results = {}
        for name, func in augmentations.items():
            try:
                output_path = self.create_augmented_video(name, func)
                results[name] = output_path
            except Exception as e:
                logger.error(f"Failed to create {name}: {e}")
        
        return results

def create_ground_truth_labels(original_labels_path: str, output_dir: str, 
                              augmentation_results: Dict[str, str]) -> None:
    """Create ground truth labels for augmented videos."""
    # Load original labels
    with open(original_labels_path, 'r') as f:
        original_labels = json.load(f)
    
    # For each augmented video, create corresponding labels
    for aug_name, video_path in augmentation_results.items():
        labels_path = Path(output_dir) / f"{Path(video_path).stem}.json"
        
        # For now, we'll use the same labels as the original
        # In a real scenario, you might need to adjust labels based on the augmentation
        with open(labels_path, 'w') as f:
            json.dump(original_labels, f, indent=4)
        
        logger.info(f"Created labels for {aug_name}: {labels_path}")

def main():
    parser = argparse.ArgumentParser(description='Augment video for blink detection testing')
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--output_dir', default='./augmented_videos', 
                       help='Output directory for augmented videos')
    parser.add_argument('--labels', help='Path to original ground truth labels')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        logger.error(f"Input video not found: {args.input}")
        return
    
    # Create augmenter and generate videos
    augmenter = VideoAugmenter(args.input, args.output_dir)
    logger.info("Starting video augmentation...")
    
    augmentation_results = augmenter.generate_all_augmentations()
    
    # Create ground truth labels if provided
    if args.labels and os.path.exists(args.labels):
        create_ground_truth_labels(args.labels, args.output_dir, augmentation_results)
    
    # Save augmentation metadata
    metadata = {
        "original_video": args.input,
        "augmentations": augmentation_results,
        "total_generated": len(augmentation_results)
    }
    
    metadata_path = Path(args.output_dir) / "augmentation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Augmentation complete! Generated {len(augmentation_results)} videos.")
    logger.info(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main() 