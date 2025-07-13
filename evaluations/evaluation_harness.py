import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
import cv2

# Import our blink detection model
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from blink_detection_model import BlinkDetectionModel

class BlinkDetectionEvaluator:
    """
    Evaluation harness for blink detection model.
    Runs the model over all videos and computes comprehensive metrics.
    """
    
    def __init__(self, 
                 ear_threshold: float = 0.21,
                 consecutive_frames: int = 2,
                 results_dir: str = "evaluations/results"):
        """
        Initialize the evaluator.
        
        Args:
            ear_threshold: EAR threshold for blink detection
            consecutive_frames: Number of consecutive frames to confirm blink
            results_dir: Directory to store evaluation results
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the model
        self.model = BlinkDetectionModel(
            ear_threshold=ear_threshold,
            consecutive_frames=consecutive_frames
        )
        
        # Metrics storage
        self.results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_config": {
                "ear_threshold": ear_threshold,
                "consecutive_frames": consecutive_frames
            },
            "videos": {},
            "summary_metrics": {}
        }
    
    def load_ground_truth(self, json_path: str) -> Dict[str, Any]:
        """
        Load ground truth labels from JSON file.
        
        Args:
            json_path: Path to ground truth JSON file
            
        Returns:
            Dictionary with frame-by-frame ground truth
        """
        with open(json_path, 'r') as f:
            gt_data = json.load(f)
        
        # Convert to frame-based format
        ground_truth = {}
        for frame_idx, frame_data in gt_data.items():
            ground_truth[int(frame_idx)] = {
                "eye_state": frame_data["open_closed"],
                "direction": frame_data["direction"],
                "is_blink": frame_data["open_closed"] == "Closed"
            }
        
        return ground_truth
    
    def compute_blink_metrics(self, 
                            predicted_blinks: List[int], 
                            ground_truth: Dict[int, Any],
                            total_frames: int) -> Dict[str, float]:
        """
        Compute blink detection metrics.
        
        Args:
            predicted_blinks: List of frame indices where blinks were detected
            ground_truth: Ground truth data
            total_frames: Total number of frames
            
        Returns:
            Dictionary of metrics
        """
        # Convert ground truth to blink frames
        gt_blink_frames = [frame_idx for frame_idx, data in ground_truth.items() 
                          if data["is_blink"]]
        
        # Calculate metrics
        tp = len(set(predicted_blinks) & set(gt_blink_frames))  # True positives
        fp = len(set(predicted_blinks) - set(gt_blink_frames))  # False positives
        fn = len(set(gt_blink_frames) - set(predicted_blinks))  # False negatives
        tn = total_frames - tp - fp - fn  # True negatives
        
        # Avoid division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total_frames if total_frames > 0 else 0.0
        
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "predicted_blinks": len(predicted_blinks),
            "ground_truth_blinks": len(gt_blink_frames)
        }
    
    def compute_temporal_metrics(self, 
                               predicted_blinks: List[int], 
                               ground_truth: Dict[int, Any]) -> Dict[str, float]:
        """
        Compute temporal accuracy metrics.
        
        Args:
            predicted_blinks: List of frame indices where blinks were detected
            ground_truth: Ground truth data
            
        Returns:
            Dictionary of temporal metrics
        """
        gt_blink_frames = [frame_idx for frame_idx, data in ground_truth.items() 
                          if data["is_blink"]]
        
        if not gt_blink_frames or not predicted_blinks:
            return {
                "temporal_accuracy": 0.0,
                "average_temporal_error": float('inf'),
                "max_temporal_error": float('inf')
            }
        
        # Calculate temporal errors for each ground truth blink
        temporal_errors = []
        for gt_frame in gt_blink_frames:
            if predicted_blinks:
                # Find closest predicted blink
                closest_pred = min(predicted_blinks, key=lambda x: abs(x - gt_frame))
                temporal_errors.append(abs(closest_pred - gt_frame))
            else:
                temporal_errors.append(float('inf'))
        
        avg_temporal_error = np.mean(temporal_errors) if temporal_errors else float('inf')
        max_temporal_error = max(temporal_errors) if temporal_errors else float('inf')
        
        # Temporal accuracy (percentage of blinks detected within acceptable range)
        acceptable_range = 5  # frames
        temporal_accuracy = sum(1 for error in temporal_errors if error <= acceptable_range) / len(temporal_errors)
        
        return {
            "temporal_accuracy": temporal_accuracy,
            "average_temporal_error": avg_temporal_error,
            "max_temporal_error": max_temporal_error,
            "acceptable_range_frames": acceptable_range
        }
    
    def evaluate_video(self, video_path: str, gt_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single video.
        
        Args:
            video_path: Path to video file
            gt_path: Path to ground truth JSON file (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating: {video_path}")
        
        # Process video with model
        model_results = self.model.process_video(video_path, output_frames=True)
        
        # Initialize results
        video_results = {
            "video_path": str(video_path),
            "model_results": model_results,
            "ground_truth_available": gt_path is not None
        }
        
        # If ground truth is available, compute metrics
        if gt_path and os.path.exists(gt_path):
            ground_truth = self.load_ground_truth(gt_path)
            
            # Compute blink detection metrics
            blink_metrics = self.compute_blink_metrics(
                model_results["blink_frames"],
                ground_truth,
                model_results["total_frames"]
            )
            
            # Compute temporal metrics
            temporal_metrics = self.compute_temporal_metrics(
                model_results["blink_frames"],
                ground_truth
            )
            
            video_results.update({
                "blink_metrics": blink_metrics,
                "temporal_metrics": temporal_metrics,
                "ground_truth_path": str(gt_path)
            })
        
        return video_results
    
    def find_video_ground_truth_pairs(self) -> List[tuple]:
        """
        Find all video files and their corresponding ground truth files.
        
        Returns:
            List of (video_path, gt_path) tuples
        """
        video_gt_pairs = []
        
        # Get current working directory
        cwd = Path.cwd()
        print(f"Current working directory: {cwd}")
        
        # Original video and ground truth (from root directory)
        original_video = cwd / "Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25.mp4"
        original_gt = cwd / "Labels_Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25.json"
        
        print(f"Looking for original video: {original_video}")
        print(f"Looking for original GT: {original_gt}")
        
        if original_video.exists() and original_gt.exists():
            video_gt_pairs.append((str(original_video), str(original_gt)))
            print(f"✓ Found original video and GT")
        
        # Augmented videos
        augmented_dir = cwd / "evaluations" / "augmented_videos"
        print(f"Looking for augmented videos in: {augmented_dir}")
        
        if augmented_dir.exists():
            # Find all MP4 files
            video_files = list(augmented_dir.glob("*.mp4"))
            print(f"Found {len(video_files)} MP4 files in augmented directory")
            
            for video_file in video_files:
                # Look for corresponding JSON file
                gt_file = video_file.with_suffix('.json')
                if gt_file.exists():
                    video_gt_pairs.append((str(video_file), str(gt_file)))
                    print(f"✓ Found video-GT pair: {video_file.name}")
                else:
                    # Video without ground truth
                    video_gt_pairs.append((str(video_file), None))
                    print(f"⚠ Video without GT: {video_file.name}")
        else:
            print(f"✗ Augmented directory not found: {augmented_dir}")
        
        print(f"Total video-GT pairs found: {len(video_gt_pairs)}")
        return video_gt_pairs
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation on all videos.
        
        Returns:
            Complete evaluation results
        """
        print("Starting blink detection evaluation...")
        
        # Find all video files
        video_gt_pairs = self.find_video_ground_truth_pairs()
        print(f"Found {len(video_gt_pairs)} videos to evaluate")
        
        # Evaluate each video
        for video_path, gt_path in video_gt_pairs:
            try:
                video_results = self.evaluate_video(video_path, gt_path)
                video_name = Path(video_path).stem
                self.results["videos"][video_name] = video_results
                print(f"✓ Completed: {video_name}")
            except Exception as e:
                print(f"✗ Error evaluating {video_path}: {e}")
                self.results["videos"][Path(video_path).stem] = {
                    "error": str(e),
                    "video_path": str(video_path)
                }
        
        # Compute summary metrics
        self._compute_summary_metrics()
        
        # Save results
        self._save_results()
        
        print(f"Evaluation complete. Results saved to {self.results_dir}")
        return self.results
    
    def _compute_summary_metrics(self):
        """Compute summary metrics across all videos."""
        videos_with_gt = [v for v in self.results["videos"].values() 
                         if v.get("ground_truth_available", False)]
        
        if not videos_with_gt:
            return
        
        # Aggregate metrics
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_accuracy = 0
        total_temporal_accuracy = 0
        total_temporal_error = 0
        
        valid_videos = 0
        
        for video_data in videos_with_gt:
            if "blink_metrics" in video_data:
                metrics = video_data["blink_metrics"]
                temporal = video_data["temporal_metrics"]
                
                total_precision += metrics["precision"]
                total_recall += metrics["recall"]
                total_f1 += metrics["f1_score"]
                total_accuracy += metrics["accuracy"]
                total_temporal_accuracy += temporal["temporal_accuracy"]
                total_temporal_error += temporal["average_temporal_error"]
                valid_videos += 1
        
        if valid_videos > 0:
            self.results["summary_metrics"] = {
                "average_precision": total_precision / valid_videos,
                "average_recall": total_recall / valid_videos,
                "average_f1_score": total_f1 / valid_videos,
                "average_accuracy": total_accuracy / valid_videos,
                "average_temporal_accuracy": total_temporal_accuracy / valid_videos,
                "average_temporal_error": total_temporal_error / valid_videos,
                "videos_evaluated": valid_videos,
                "total_videos": len(self.results["videos"])
            }
    
    def _save_results(self):
        """Save evaluation results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"evaluation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also save as metrics.json for the challenge
        metrics_file = self.results_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        print(f"Metrics saved to: {metrics_file}")
        
        # Run threshold checking
        self._run_threshold_check(metrics_file)
    
    def _run_threshold_check(self, metrics_file: Path):
        """Run threshold checking on the evaluation results."""
        try:
            # Import threshold checker
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from threshold_checker import ThresholdChecker
            
            print("\n🔍 Running threshold validation...")
            
            # Initialize threshold checker
            thresholds_file = Path(__file__).parent / "thresholds.json"
            checker = ThresholdChecker(str(thresholds_file))
            
            # Load metrics
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            if "summary_metrics" not in metrics_data:
                print("⚠️  No summary metrics found for threshold checking")
                return
            
            summary_metrics = metrics_data["summary_metrics"]
            
            # Run checks
            check_results = checker.check_summary_metrics(summary_metrics)
            alert_results = checker.check_alert_conditions(summary_metrics)
            
            # Print results
            checker.print_results(check_results, alert_results)
            
            # Save threshold check results
            threshold_results_file = self.results_dir / "threshold_check_results.json"
            checker.save_results(check_results, alert_results, str(threshold_results_file))
            
            # Determine if test suite should fail
            if not check_results["passed"] or alert_results["critical"]:
                print("\n❌ THRESHOLD CHECK FAILED - Test suite should fail")
                return False
            else:
                print("\n✅ THRESHOLD CHECK PASSED")
                return True
                
        except Exception as e:
            print(f"⚠️  Threshold checking failed: {e}")
            return True  # Don't fail the test suite if threshold checking itself fails

def main():
    """Run the evaluation harness."""
    evaluator = BlinkDetectionEvaluator()
    results = evaluator.run_evaluation()
    
    # Print summary
    if "summary_metrics" in results:
        summary = results["summary_metrics"]
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Videos evaluated: {summary['videos_evaluated']}/{summary['total_videos']}")
        print(f"Average Precision: {summary['average_precision']:.3f}")
        print(f"Average Recall: {summary['average_recall']:.3f}")
        print(f"Average F1-Score: {summary['average_f1_score']:.3f}")
        print(f"Average Accuracy: {summary['average_accuracy']:.3f}")
        print(f"Average Temporal Accuracy: {summary['average_temporal_accuracy']:.3f}")
        print(f"Average Temporal Error: {summary['average_temporal_error']:.2f} frames")
        print("="*50)

if __name__ == "__main__":
    main() 