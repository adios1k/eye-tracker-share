# Blink Detection Evaluation Framework

This directory contains the evaluation harness for the blink detection model, designed for the CV automation challenge.

## Overview

The evaluation framework provides a comprehensive test suite that:
- Loads the blink detection model
- Runs it over all videos (original + augmented)
- Computes multiple evaluation metrics
- Stores results in structured JSON format

## Files

- `blink_detection_model.py` - Modular blink detection model class
- `evaluation_harness.py` - Main evaluation framework with integrated threshold checking
- `threshold_checker.py` - Threshold validation system
- `thresholds.json` - Acceptance thresholds for metrics
- `test_evaluation.py` - Test script demonstrating usage
- `test_threshold_failure.py` - Test script demonstrating threshold failures
- `augment_video.py` - Video augmentation script (from previous step)
- `augmented_videos/` - Directory containing augmented videos and ground truth

## Usage

### Basic Evaluation

```python
from evaluation_harness import BlinkDetectionEvaluator

# Run evaluation with default settings
evaluator = BlinkDetectionEvaluator()
results = evaluator.run_evaluation()
```

### Custom Configuration

```python
# Customize model parameters
evaluator = BlinkDetectionEvaluator(
    ear_threshold=0.25,        # EAR threshold for blink detection
    consecutive_frames=3,       # Frames required to confirm blink
    results_dir="custom_results"
)
results = evaluator.run_evaluation()
```

### Single Video Evaluation

```python
# Evaluate a single video
result = evaluator.evaluate_video(
    video_path="path/to/video.mp4",
    gt_path="path/to/ground_truth.json"
)
```

## Metrics Computed

### Blink Detection Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: (True positives + True negatives) / Total frames
- **True Positives**: Correctly detected blinks
- **False Positives**: Incorrectly detected blinks
- **False Negatives**: Missed blinks
- **True Negatives**: Correctly identified non-blink frames

### Temporal Metrics
- **Temporal Accuracy**: Percentage of blinks detected within acceptable range (5 frames)
- **Average Temporal Error**: Average frame difference between predicted and ground truth blinks
- **Max Temporal Error**: Maximum frame difference for any blink
- **Acceptable Range**: 5 frames (configurable)

### Model Performance Metrics
- **Face Detection Rate**: Percentage of frames where face was detected
- **Blink Rate**: Percentage of frames with detected blinks
- **Average EAR**: Average Eye Aspect Ratio across all frames

## Results Structure

Results are saved in `evaluations/results/metrics.json` with the following structure:

```json
{
  "evaluation_timestamp": "2025-07-13T18:00:01.026663",
  "model_config": {
    "ear_threshold": 0.21,
    "consecutive_frames": 2
  },
  "videos": {
    "video_name": {
      "video_path": "path/to/video.mp4",
      "model_results": {
        "total_frames": 2462,
        "frames_with_face": 2462,
        "frames_with_blink": 1210,
        "blink_frames": [41, 42, 43, ...],
        "average_ear": 0.207,
        "face_detection_rate": 1.0,
        "blink_rate": 0.487
      },
      "ground_truth_available": true,
      "blink_metrics": {
        "precision": 0.481,
        "recall": 0.909,
        "f1_score": 0.629,
        "accuracy": 0.723,
        "true_positives": 577,
        "false_positives": 623,
        "false_negatives": 58,
        "true_negatives": 1204
      },
      "temporal_metrics": {
        "temporal_accuracy": 0.995,
        "average_temporal_error": 0.231,
        "max_temporal_error": 23,
        "acceptable_range_frames": 5
      }
    }
  },
  "summary_metrics": {
    "average_precision": 0.473,
    "average_recall": 0.893,
    "average_f1_score": 0.618,
    "average_accuracy": 0.716,
    "average_temporal_accuracy": 0.986,
    "average_temporal_error": 0.494,
    "videos_evaluated": 10,
    "total_videos": 12
  }
}
```

## Running the Evaluation

### From Command Line

```bash
# From project root
python evaluations/evaluation_harness.py

# From evaluations directory
cd evaluations
python evaluation_harness.py
```

### Test Script

```bash
# Run comprehensive tests
python evaluations/test_evaluation.py
```

## Threshold System

The evaluation framework includes a comprehensive threshold system that validates metrics against defined acceptance criteria.

### Threshold Configuration

Thresholds are defined in `thresholds.json`:

```json
{
  "thresholds": {
    "blink_detection": {
      "precision": {"minimum": 0.4, "target": 0.6},
      "recall": {"minimum": 0.8, "target": 0.9},
      "f1_score": {"minimum": 0.5, "target": 0.7},
      "accuracy": {"minimum": 0.6, "target": 0.8}
    },
    "temporal_accuracy": {
      "temporal_accuracy": {"minimum": 0.9, "target": 0.95},
      "average_temporal_error": {"maximum": 2.0, "target": 1.0}
    },
    "model_performance": {
      "face_detection_rate": {"minimum": 0.95, "target": 0.98},
      "videos_evaluated": {"minimum": 8, "target": 10}
    }
  }
}
```

### Alert System

The system includes three levels of alerts:

- **Critical**: Failures that should stop the pipeline
- **Warning**: Issues that indicate potential problems
- **Info**: Informational alerts for monitoring

### Running Threshold Checks

```bash
# Check thresholds against existing metrics
python evaluations/threshold_checker.py --metrics results/metrics.json

# Test threshold failure scenarios
python evaluations/test_threshold_failure.py
```

### Threshold Check Results

The system automatically:
- Validates all metrics against thresholds
- Generates detailed reports
- Fails the test suite when metrics fall below acceptable levels
- Saves results to `threshold_check_results.json`

## Current Results

Based on the latest evaluation run:

- **Videos Evaluated**: 10/12 (2 videos had file access issues)
- **Average Precision**: 0.473
- **Average Recall**: 0.893
- **Average F1-Score**: 0.618
- **Average Accuracy**: 0.716
- **Average Temporal Accuracy**: 0.986
- **Average Temporal Error**: 0.49 frames

## Supported Video Formats

- MP4 files with corresponding JSON ground truth files
- Videos are automatically paired with ground truth files
- Ground truth files should contain frame-by-frame annotations

## Error Handling

The framework handles various error conditions:
- Missing video files
- Corrupted video files
- Missing ground truth files
- Model processing errors

Errors are logged and included in the results without stopping the evaluation.

## Configuration

Key parameters that can be adjusted:

- `ear_threshold`: Eye Aspect Ratio threshold (default: 0.21)
- `consecutive_frames`: Frames required to confirm blink (default: 2)
- `min_detection_confidence`: MediaPipe face detection confidence (default: 0.5)
- `min_tracking_confidence`: MediaPipe face tracking confidence (default: 0.5)

## Dependencies

- OpenCV
- MediaPipe
- NumPy
- Pathlib (standard library)
- JSON (standard library)

## Notes

- The evaluation framework is designed to be robust and handle various edge cases
- Results are automatically saved with timestamps
- The framework can be extended with additional metrics as needed
- All metrics are computed per-video and aggregated across all videos 