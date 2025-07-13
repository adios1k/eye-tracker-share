#!/usr/bin/env python3
"""
Test script to demonstrate threshold failure scenarios.
This script shows how the threshold system fails when metrics fall below acceptable levels.
"""

import json
import sys
from pathlib import Path

# Add the evaluations directory to the path
sys.path.append(Path(__file__).parent)

from threshold_checker import ThresholdChecker

def create_failing_metrics():
    """Create metrics that would fail threshold checks."""
    return {
        "average_precision": 0.35,  # Below minimum of 0.4
        "average_recall": 0.75,     # Below minimum of 0.8
        "average_f1_score": 0.45,   # Below minimum of 0.5
        "average_accuracy": 0.55,   # Below minimum of 0.6
        "average_temporal_accuracy": 0.85,  # Below minimum of 0.9
        "average_temporal_error": 3.5,      # Above maximum of 2.0
        "videos_evaluated": 6,      # Below minimum of 8
        "total_videos": 12
    }

def create_warning_metrics():
    """Create metrics that would trigger warnings but not failures."""
    return {
        "average_precision": 0.45,  # Below target but above minimum
        "average_recall": 0.82,     # Below target but above minimum
        "average_f1_score": 0.55,   # Below target but above minimum
        "average_accuracy": 0.65,   # Below target but above minimum
        "average_temporal_accuracy": 0.92,  # Below target but above minimum
        "average_temporal_error": 1.8,      # Above target but below maximum
        "videos_evaluated": 9,      # Below target but above minimum
        "total_videos": 12
    }

def test_threshold_failures():
    """Test threshold failure scenarios."""
    
    print("Testing Threshold Failure Scenarios")
    print("=" * 50)
    
    # Test 1: Complete failure scenario
    print("\n1. Testing complete failure scenario...")
    failing_metrics = create_failing_metrics()
    
    checker = ThresholdChecker()
    check_results = checker.check_summary_metrics(failing_metrics)
    alert_results = checker.check_alert_conditions(failing_metrics)
    
    print(f"Expected: FAILED")
    print(f"Actual: {'FAILED' if not check_results['passed'] else 'PASSED'}")
    print(f"Violations: {len(check_results['violations'])}")
    print(f"Critical Alerts: {len(alert_results['critical'])}")
    print(f"Warnings: {len(alert_results['warning'])}")
    
    # Test 2: Warning scenario
    print("\n2. Testing warning scenario...")
    warning_metrics = create_warning_metrics()
    
    check_results2 = checker.check_summary_metrics(warning_metrics)
    alert_results2 = checker.check_alert_conditions(warning_metrics)
    
    print(f"Expected: PASSED with warnings")
    print(f"Actual: {'PASSED' if check_results2['passed'] else 'FAILED'}")
    print(f"Violations: {len(check_results2['violations'])}")
    print(f"Critical Alerts: {len(alert_results2['critical'])}")
    print(f"Warnings: {len(alert_results2['warning'])}")
    
    # Test 3: Success scenario (using real metrics)
    print("\n3. Testing success scenario...")
    try:
        with open("results/metrics.json", 'r') as f:
            real_metrics = json.load(f)
        
        if "summary_metrics" in real_metrics:
            real_summary = real_metrics["summary_metrics"]
            check_results3 = checker.check_summary_metrics(real_summary)
            alert_results3 = checker.check_alert_conditions(real_summary)
            
            print(f"Expected: PASSED")
            print(f"Actual: {'PASSED' if check_results3['passed'] else 'FAILED'}")
            print(f"Violations: {len(check_results3['violations'])}")
            print(f"Critical Alerts: {len(alert_results3['critical'])}")
            print(f"Warnings: {len(alert_results3['warning'])}")
        else:
            print("No real metrics available for testing")
    except FileNotFoundError:
        print("No real metrics file available for testing")

def test_individual_thresholds():
    """Test individual threshold checks."""
    
    print("\n" + "=" * 50)
    print("Testing Individual Threshold Checks")
    print("=" * 50)
    
    checker = ThresholdChecker()
    
    # Test precision threshold
    print("\nTesting precision threshold (minimum: 0.4):")
    test_values = [0.35, 0.4, 0.45, 0.5]
    for value in test_values:
        passed, message = checker.check_metric("precision", value, {"minimum": 0.4})
        print(f"  {message}")
    
    # Test temporal error threshold
    print("\nTesting temporal error threshold (maximum: 2.0):")
    test_values = [1.5, 2.0, 2.5, 3.0]
    for value in test_values:
        passed, message = checker.check_metric("temporal_error", value, {"maximum": 2.0})
        print(f"  {message}")
    
    # Test videos evaluated threshold
    print("\nTesting videos evaluated threshold (minimum: 8):")
    test_values = [6, 8, 10, 12]
    for value in test_values:
        passed, message = checker.check_metric("videos_evaluated", value, {"minimum": 8})
        print(f"  {message}")

def test_alert_conditions():
    """Test alert condition evaluation."""
    
    print("\n" + "=" * 50)
    print("Testing Alert Conditions")
    print("=" * 50)
    
    checker = ThresholdChecker()
    
    # Test critical conditions
    print("\nTesting critical conditions:")
    test_metrics = {
        "face_detection_rate": 0.85,  # Should trigger critical alert
        "videos_evaluated": 3,        # Should trigger critical alert
        "failed_videos": 6            # Should trigger critical alert
    }
    
    alert_results = checker.check_alert_conditions(test_metrics)
    print(f"Critical alerts: {alert_results['critical']}")
    
    # Test warning conditions
    print("\nTesting warning conditions:")
    test_metrics = {
        "precision": 0.45,            # Should trigger warning
        "recall": 0.82,              # Should trigger warning
        "f1_score": 0.55,            # Should trigger warning
        "temporal_accuracy": 0.92    # Should trigger warning
    }
    
    alert_results = checker.check_alert_conditions(test_metrics)
    print(f"Warning alerts: {alert_results['warning']}")

if __name__ == "__main__":
    test_threshold_failures()
    test_individual_thresholds()
    test_alert_conditions()
    
    print("\n" + "=" * 50)
    print("THRESHOLD TESTING COMPLETE")
    print("=" * 50) 