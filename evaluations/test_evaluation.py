#!/usr/bin/env python3
"""
Test script for the blink detection evaluation harness.
This script demonstrates how to use the evaluation framework.
"""

import sys
import os
from pathlib import Path

# Add the evaluations directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation_harness import BlinkDetectionEvaluator

def test_evaluation():
    """Test the evaluation harness with different configurations."""
    
    print("Testing Blink Detection Evaluation Harness")
    print("=" * 50)
    
    # Test 1: Default configuration
    print("\n1. Testing with default configuration...")
    evaluator = BlinkDetectionEvaluator()
    results = evaluator.run_evaluation()
    
    # Test 2: Different EAR threshold
    print("\n2. Testing with different EAR threshold (0.25)...")
    evaluator2 = BlinkDetectionEvaluator(ear_threshold=0.25)
    results2 = evaluator2.run_evaluation()
    
    # Test 3: Different consecutive frames
    print("\n3. Testing with different consecutive frames (3)...")
    evaluator3 = BlinkDetectionEvaluator(consecutive_frames=3)
    results3 = evaluator3.run_evaluation()
    
    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON OF DIFFERENT CONFIGURATIONS")
    print("=" * 50)
    
    configs = [
        ("Default (EAR=0.21, frames=2)", results),
        ("Higher EAR (0.25, frames=2)", results2),
        ("More frames (EAR=0.21, frames=3)", results3)
    ]
    
    for name, result in configs:
        if "summary_metrics" in result:
            metrics = result["summary_metrics"]
            print(f"\n{name}:")
            print(f"  Precision: {metrics.get('average_precision', 0):.3f}")
            print(f"  Recall: {metrics.get('average_recall', 0):.3f}")
            print(f"  F1-Score: {metrics.get('average_f1_score', 0):.3f}")
            print(f"  Accuracy: {metrics.get('average_accuracy', 0):.3f}")
            print(f"  Temporal Accuracy: {metrics.get('average_temporal_accuracy', 0):.3f}")
            print(f"  Videos evaluated: {metrics.get('videos_evaluated', 0)}")
    
    print("\n" + "=" * 50)
    print("EVALUATION FRAMEWORK TEST COMPLETE")
    print("=" * 50)

def test_single_video():
    """Test evaluation on a single video."""
    
    print("\nTesting single video evaluation...")
    
    evaluator = BlinkDetectionEvaluator()
    
    # Test with original video
    video_path = "Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25.mp4"
    gt_path = "Labels_Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25.json"
    
    if os.path.exists(video_path) and os.path.exists(gt_path):
        print(f"Evaluating single video: {video_path}")
        result = evaluator.evaluate_video(video_path, gt_path)
        
        if "blink_metrics" in result:
            metrics = result["blink_metrics"]
            print(f"Single video results:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Predicted blinks: {metrics['predicted_blinks']}")
            print(f"  Ground truth blinks: {metrics['ground_truth_blinks']}")
    else:
        print(f"Video or ground truth file not found")

if __name__ == "__main__":
    test_evaluation()
    test_single_video() 