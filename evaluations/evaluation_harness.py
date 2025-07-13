import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

# Import our blink detection model
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .blink_detection_model import BlinkDetectionModel

# Import advanced evaluation modules
try:
    from .advanced_metrics import calculate_advanced_metrics
    from .llm_summarizer import generate_llm_summary
    from .advanced_augmentation import create_advanced_augmentation_pipeline, AugmentationConfig
    advanced_features_available = True
except ImportError as e:
    print(f"Warning: Advanced features not available: {e}")
    advanced_features_available = False

class BlinkDetectionEvaluator:
    """
    Evaluation harness for blink detection model.
    Runs the model over all videos and computes comprehensive metrics.
    """
    
    def __init__(self, 
                 ear_threshold: float = 0.21,
                 consecutive_frames: int = 2,
                 results_dir: str = "evaluations/results") -> None:
        """
        Initialize the evaluator.
        
        Args:
            ear_threshold: EAR threshold for blink detection
            consecutive_frames: Number of consecutive frames to confirm blink
            results_dir: Directory to store evaluation results
        """
        self.ear_threshold: float = ear_threshold
        self.consecutive_frames: int = consecutive_frames
        self.results_dir: Path = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the model
        self.model: BlinkDetectionModel = BlinkDetectionModel(
            ear_threshold=ear_threshold,
            consecutive_frames=consecutive_frames
        )
        
        # Metrics storage
        self.results: Dict[str, Any] = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_config": {
                "ear_threshold": ear_threshold,
                "consecutive_frames": consecutive_frames
            },
            "videos": {},
            "summary_metrics": {}
        }
    
    def load_ground_truth(self, json_path: str) -> Dict[int, Dict[str, Any]]:
        """
        Load ground truth labels from JSON file.
        
        Args:
            json_path: Path to ground truth JSON file
            
        Returns:
            Dictionary with frame-by-frame ground truth
        """
        with open(json_path, 'r') as f:
            gt_data: Dict[str, Any] = json.load(f)
        
        # Convert to frame-based format
        ground_truth: Dict[int, Dict[str, Any]] = {}
        for frame_idx_str, frame_data in gt_data.items():
            frame_idx = int(frame_idx_str)
            ground_truth[frame_idx] = {
                "eye_state": frame_data["open_closed"],
                "direction": frame_data["direction"],
                "is_blink": frame_data["open_closed"] == "Closed"
            }
        
        return ground_truth
    
    def compute_blink_metrics(self, 
                            predicted_blinks: List[int], 
                            ground_truth: Dict[int, Dict[str, Any]],
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
            "true_positives": float(tp),
            "false_positives": float(fp),
            "false_negatives": float(fn),
            "true_negatives": float(tn),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "accuracy": float(accuracy),
            "predicted_blinks": float(len(predicted_blinks)),
            "ground_truth_blinks": float(len(gt_blink_frames))
        }
    
    def compute_temporal_metrics(self, 
                               predicted_blinks: List[int], 
                               ground_truth: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
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
        temporal_errors: List[float] = []
        for gt_frame in gt_blink_frames:
            if predicted_blinks:
                # Find closest predicted blink
                closest_pred = min(predicted_blinks, key=lambda x: abs(x - gt_frame))
                temporal_errors.append(float(abs(closest_pred - gt_frame)))
            else:
                temporal_errors.append(float('inf'))
        
        avg_temporal_error = np.mean(temporal_errors) if temporal_errors else float('inf')
        max_temporal_error = max(temporal_errors) if temporal_errors else float('inf')
        
        # Temporal accuracy (percentage of blinks detected within acceptable range)
        acceptable_range = 5  # frames
        temporal_accuracy = sum(1 for error in temporal_errors if error <= acceptable_range) / len(temporal_errors)
        
        return {
            "temporal_accuracy": float(temporal_accuracy),
            "average_temporal_error": float(avg_temporal_error),
            "max_temporal_error": float(max_temporal_error),
            "acceptable_range_frames": float(acceptable_range)
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
        model_results: Dict[str, Any] = self.model.process_video(video_path, output_frames=True)
        
        # Initialize results
        video_results: Dict[str, Any] = {
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
    
    def find_video_ground_truth_pairs(self) -> List[Tuple[str, str]]:
        """
        Find all video files and their corresponding ground truth files.
        
        Returns:
            List of (video_path, gt_path) tuples
        """
        video_gt_pairs: List[Tuple[str, str]] = []
        
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
                json_file = video_file.with_suffix('.json')
                if json_file.exists():
                    video_gt_pairs.append((str(video_file), str(json_file)))
                    print(f"✓ Found pair: {video_file.name} + {json_file.name}")
                else:
                    print(f"⚠ No ground truth found for: {video_file.name}")
        
        return video_gt_pairs
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete evaluation suite.
        
        Returns:
            Dictionary with all evaluation results
        """
        print("Starting blink detection evaluation...")
        
        # Find all video-ground truth pairs
        video_gt_pairs = self.find_video_ground_truth_pairs()
        
        if not video_gt_pairs:
            print("❌ No video-ground truth pairs found!")
            return self.results
        
        print(f"Found {len(video_gt_pairs)} video-ground truth pairs")
        
        # Evaluate each video
        for video_path, gt_path in video_gt_pairs:
            try:
                video_results = self.evaluate_video(video_path, gt_path)
                video_name = Path(video_path).stem
                self.results["videos"][video_name] = video_results
                print(f"✅ Completed evaluation for: {video_name}")
            except Exception as e:
                print(f"❌ Error evaluating {video_path}: {e}")
                video_name = Path(video_path).stem
                self.results["videos"][video_name] = {
                    "error": str(e),
                    "video_path": str(video_path),
                    "gt_path": str(gt_path)
                }
        
        # Compute summary metrics
        self._compute_summary_metrics()
        
        # Save results
        self._save_results()
        
        # Run threshold check
        metrics_file = self.results_dir / "metrics.json"
        self._run_threshold_check(metrics_file)
        
        return self.results
    
    def _compute_summary_metrics(self) -> None:
        """Compute summary metrics across all videos."""
        videos_with_gt = [v for v in self.results["videos"].values() 
                         if isinstance(v, dict) and v.get("ground_truth_available", False)]
        
        if not videos_with_gt:
            print("⚠ No videos with ground truth found for summary metrics")
            return
        
        # Collect all metrics
        precisions: List[float] = []
        recalls: List[float] = []
        f1_scores: List[float] = []
        accuracies: List[float] = []
        temporal_accuracies: List[float] = []
        temporal_errors: List[float] = []
        
        for video_data in videos_with_gt:
            if "blink_metrics" in video_data:
                metrics = video_data["blink_metrics"]
                precisions.append(metrics["precision"])
                recalls.append(metrics["recall"])
                f1_scores.append(metrics["f1_score"])
                accuracies.append(metrics["accuracy"])
            
            if "temporal_metrics" in video_data:
                temporal = video_data["temporal_metrics"]
                temporal_accuracies.append(temporal["temporal_accuracy"])
                if temporal["average_temporal_error"] != float('inf'):
                    temporal_errors.append(temporal["average_temporal_error"])
        
        # Compute averages
        summary_metrics = {
            "videos_evaluated": len(videos_with_gt),
            "total_videos": len(self.results["videos"]),
            "average_precision": float(np.mean(precisions)) if precisions else 0.0,
            "average_recall": float(np.mean(recalls)) if recalls else 0.0,
            "average_f1_score": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "average_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "average_temporal_accuracy": float(np.mean(temporal_accuracies)) if temporal_accuracies else 0.0,
            "average_temporal_error": float(np.mean(temporal_errors)) if temporal_errors else float('inf'),
            "max_temporal_error": float(max(temporal_errors)) if temporal_errors else float('inf')
        }
        
        self.results["summary_metrics"] = summary_metrics
        print(f"📊 Summary metrics computed for {len(videos_with_gt)} videos")
        
        # Add advanced metrics if available
        if advanced_features_available:
            self._compute_advanced_metrics(videos_with_gt)
    
    def _compute_advanced_metrics(self, videos_with_gt: List[Dict[str, Any]]) -> None:
        """Compute advanced metrics using the advanced metrics module."""
        try:
            # Prepare data for advanced metrics
            predictions = []
            ground_truth = []
            
            for video_data in videos_with_gt:
                if "model_results" in video_data and "ground_truth" in video_data:
                    model_results = video_data["model_results"]
                    gt_data = video_data["ground_truth"]
                    
                    # Convert to prediction format
                    for frame_idx in range(model_results["total_frames"]):
                        is_blink = frame_idx in model_results["blink_frames"]
                        confidence = 0.8 if is_blink else 0.2  # Simple confidence model
                        
                        predictions.append({
                            "frame_idx": frame_idx,
                            "blink_detected": is_blink,
                            "confidence": confidence,
                            "timestamp": frame_idx / 30.0  # Assuming 30 fps
                        })
                        
                        # Ground truth
                        gt_frame = gt_data.get(frame_idx, {})
                        ground_truth.append({
                            "frame_idx": frame_idx,
                            "blink_detected": gt_frame.get("is_blink", False),
                            "timestamp": frame_idx / 30.0
                        })
            
            if predictions and ground_truth:
                # Calculate advanced metrics
                advanced_metrics = calculate_advanced_metrics(
                    predictions=predictions,
                    ground_truth=ground_truth,
                    video_metadata={"fps": 30, "resolution": "HD"}
                )
                
                # Store advanced metrics
                self.results["advanced_metrics"] = advanced_metrics
                
                # Generate LLM summary
                summary, dashboard_path = generate_llm_summary(
                    metrics=advanced_metrics,
                    historical_data=None  # Could be loaded from previous evaluations
                )
                
                self.results["llm_summary"] = {
                    "overall_score": summary.overall_score,
                    "insights": [{"category": i.category, "title": i.title, "description": i.description, "severity": i.severity} for i in summary.insights],
                    "recommendations": summary.recommendations,
                    "dashboard_path": dashboard_path
                }
                
                print(f"Advanced metrics computed and LLM summary generated: {dashboard_path}")
                
        except Exception as e:
            print(f"Error computing advanced metrics: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_results(self) -> None:
        """Save evaluation results to files."""
        # Save detailed results
        results_file = self.results_dir / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save metrics summary
        metrics_file = self.results_dir / "metrics.json"
        metrics_data = {
            "evaluation_timestamp": self.results["evaluation_timestamp"],
            "model_config": self.results["model_config"],
            "summary_metrics": self.results["summary_metrics"],
            "videos": {
                name: {
                    "blink_metrics": data.get("blink_metrics", {}),
                    "temporal_metrics": data.get("temporal_metrics", {}),
                    "ground_truth_available": data.get("ground_truth_available", False)
                }
                for name, data in self.results["videos"].items()
                if isinstance(data, dict)
            }
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"💾 Results saved to: {self.results_dir}")
    
    def _run_threshold_check(self, metrics_file: Path) -> None:
        """Run threshold checking on the metrics."""
        try:
            from .threshold_checker import ThresholdChecker
            
            # Load metrics
            with open(metrics_file, 'r') as f:
                metrics_data: Dict[str, Any] = json.load(f)
            
            summary_metrics = metrics_data["summary_metrics"]
            
            # Initialize threshold checker
            thresholds_file = Path(__file__).parent / "thresholds.json"
            checker = ThresholdChecker(str(thresholds_file))
            
            # Run threshold check
            check_results = checker.check_summary_metrics(summary_metrics)
            alert_results = checker.check_alert_conditions(summary_metrics)
            
            # Combine results
            threshold_results = {
                "threshold_check": check_results,
                "alerts": alert_results
            }
            
            # Save threshold check results
            threshold_file = self.results_dir / "threshold_check_results.json"
            with open(threshold_file, 'w') as f:
                json.dump(threshold_results, f, indent=2)
            
            if check_results["passed"]:
                print("✅ Threshold check passed")
            else:
                print("❌ Threshold check failed")
                violations = check_results["violations"]
                print(f"   Violations: {len(violations)}")
                for violation in violations:
                    print(f"   - {violation}")
        
        except ImportError:
            print("⚠ Threshold checker not available, skipping threshold check")
        except Exception as e:
            print(f"⚠ Error running threshold check: {e}")

def main() -> None:
    """Main evaluation function."""
    evaluator = BlinkDetectionEvaluator()
    results = evaluator.run_evaluation()
    
    # Print summary
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