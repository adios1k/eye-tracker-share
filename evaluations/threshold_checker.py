#!/usr/bin/env python3
"""
Threshold checker for blink detection evaluation.
Validates metrics against defined thresholds and fails the test suite when metrics fall below acceptable levels.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

class ThresholdChecker:
    """
    Validates evaluation results against defined thresholds.
    """
    
    def __init__(self, thresholds_file: str = "thresholds.json"):
        """
        Initialize the threshold checker.
        
        Args:
            thresholds_file: Path to thresholds.json file
        """
        self.thresholds_file = Path(thresholds_file)
        self.thresholds = self._load_thresholds()
        self.violations = []
        self.warnings = []
        self.info_alerts = []
        
    def _load_thresholds(self) -> Dict[str, Any]:
        """Load thresholds from JSON file."""
        if not self.thresholds_file.exists():
            raise FileNotFoundError(f"Thresholds file not found: {self.thresholds_file}")
        
        with open(self.thresholds_file, 'r') as f:
            return json.load(f)
    
    def check_metric(self, 
                    metric_name: str, 
                    metric_value: float, 
                    threshold_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check a single metric against its threshold.
        
        Args:
            metric_name: Name of the metric
            metric_value: Actual value of the metric
            threshold_config: Threshold configuration for this metric
            
        Returns:
            Tuple of (passed, message)
        """
        passed = True
        message = f"✓ {metric_name}: {metric_value:.3f}"
        
        # Check minimum threshold
        if "minimum" in threshold_config:
            if metric_value < threshold_config["minimum"]:
                passed = False
                message = f"✗ {metric_name}: {metric_value:.3f} < {threshold_config['minimum']} (minimum)"
        
        # Check maximum threshold
        if "maximum" in threshold_config:
            if metric_value > threshold_config["maximum"]:
                passed = False
                message = f"✗ {metric_name}: {metric_value:.3f} > {threshold_config['maximum']} (maximum)"
        
        return passed, message
    
    def check_summary_metrics(self, summary_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check summary metrics against thresholds.
        
        Args:
            summary_metrics: Summary metrics from evaluation results
            
        Returns:
            Dictionary with check results
        """
        results = {
            "passed": True,
            "violations": [],
            "warnings": [],
            "info_alerts": [],
            "details": {}
        }
        
        # Check blink detection metrics
        if "blink_detection" in self.thresholds["thresholds"]:
            for metric, config in self.thresholds["thresholds"]["blink_detection"].items():
                if metric in summary_metrics:
                    passed, message = self.check_metric(metric, summary_metrics[metric], config)
                    results["details"][metric] = {
                        "passed": passed,
                        "message": message,
                        "value": summary_metrics[metric],
                        "threshold": config
                    }
                    if not passed:
                        results["passed"] = False
                        results["violations"].append(message)
        
        # Check temporal accuracy metrics
        if "temporal_accuracy" in self.thresholds["thresholds"]:
            for metric, config in self.thresholds["thresholds"]["temporal_accuracy"].items():
                if metric in summary_metrics:
                    passed, message = self.check_metric(metric, summary_metrics[metric], config)
                    results["details"][metric] = {
                        "passed": passed,
                        "message": message,
                        "value": summary_metrics[metric],
                        "threshold": config
                    }
                    if not passed:
                        results["passed"] = False
                        results["violations"].append(message)
        
        # Check model performance metrics
        if "model_performance" in self.thresholds["thresholds"]:
            for metric, config in self.thresholds["thresholds"]["model_performance"].items():
                if metric in summary_metrics:
                    passed, message = self.check_metric(metric, summary_metrics[metric], config)
                    results["details"][metric] = {
                        "passed": passed,
                        "message": message,
                        "value": summary_metrics[metric],
                        "threshold": config
                    }
                    if not passed:
                        results["passed"] = False
                        results["violations"].append(message)
        
        # Check robustness metrics
        if "robustness" in self.thresholds["thresholds"]:
            for metric, config in self.thresholds["thresholds"]["robustness"].items():
                if metric in summary_metrics:
                    passed, message = self.check_metric(metric, summary_metrics[metric], config)
                    results["details"][metric] = {
                        "passed": passed,
                        "message": message,
                        "value": summary_metrics[metric],
                        "threshold": config
                    }
                    if not passed:
                        results["passed"] = False
                        results["violations"].append(message)
        
        return results
    
    def check_alert_conditions(self, summary_metrics: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Check for alert conditions based on thresholds.
        
        Args:
            summary_metrics: Summary metrics from evaluation results
            
        Returns:
            Dictionary with alert conditions
        """
        alerts = {
            "critical": [],
            "warning": [],
            "info": []
        }
        
        # Check critical conditions
        for condition in self.thresholds["alerts"]["critical"]["conditions"]:
            if self._evaluate_condition(condition, summary_metrics):
                alerts["critical"].append(condition)
        
        # Check warning conditions
        for condition in self.thresholds["alerts"]["warning"]["conditions"]:
            if self._evaluate_condition(condition, summary_metrics):
                alerts["warning"].append(condition)
        
        # Check info conditions
        for condition in self.thresholds["alerts"]["info"]["conditions"]:
            if self._evaluate_condition(condition, summary_metrics):
                alerts["info"].append(condition)
        
        return alerts
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """
        Evaluate a condition string against metrics.
        
        Args:
            condition: Condition string like "precision < 0.5"
            metrics: Dictionary of metrics
            
        Returns:
            True if condition is met
        """
        try:
            # Parse condition like "precision < 0.5"
            parts = condition.split()
            if len(parts) != 3:
                return False
            
            metric_name = parts[0]
            operator = parts[1]
            threshold_value = float(parts[2])
            
            if metric_name not in metrics:
                return False
            
            metric_value = metrics[metric_name]
            
            if operator == "<":
                return metric_value < threshold_value
            elif operator == ">":
                return metric_value > threshold_value
            elif operator == "<=":
                return metric_value <= threshold_value
            elif operator == ">=":
                return metric_value >= threshold_value
            elif operator == "==":
                return metric_value == threshold_value
            elif operator == "!=":
                return metric_value != threshold_value
            else:
                return False
                
        except (ValueError, IndexError):
            return False
    
    def print_results(self, check_results: Dict[str, Any], alert_results: Dict[str, List[str]]):
        """
        Print threshold check results in a formatted way.
        
        Args:
            check_results: Results from threshold checking
            alert_results: Results from alert condition checking
        """
        print("\n" + "="*60)
        print("THRESHOLD VALIDATION RESULTS")
        print("="*60)
        
        # Print metric check results
        print("\n📊 METRIC CHECKS:")
        for metric, details in check_results["details"].items():
            status = "✓ PASS" if details["passed"] else "✗ FAIL"
            print(f"  {status} - {details['message']}")
        
        # Print violations
        if check_results["violations"]:
            print(f"\n🚨 VIOLATIONS ({len(check_results['violations'])}):")
            for violation in check_results["violations"]:
                print(f"  • {violation}")
        
        # Print alerts
        if alert_results["critical"]:
            print(f"\n🔥 CRITICAL ALERTS ({len(alert_results['critical'])}):")
            for alert in alert_results["critical"]:
                print(f"  • {alert}")
        
        if alert_results["warning"]:
            print(f"\n⚠️  WARNINGS ({len(alert_results['warning'])}):")
            for alert in alert_results["warning"]:
                print(f"  • {alert}")
        
        if alert_results["info"]:
            print(f"\nℹ️  INFO ALERTS ({len(alert_results['info'])}):")
            for alert in alert_results["info"]:
                print(f"  • {alert}")
        
        # Print summary
        print(f"\n📋 SUMMARY:")
        print(f"  Overall Status: {'✓ PASSED' if check_results['passed'] else '✗ FAILED'}")
        print(f"  Violations: {len(check_results['violations'])}")
        print(f"  Critical Alerts: {len(alert_results['critical'])}")
        print(f"  Warnings: {len(alert_results['warning'])}")
        print(f"  Info Alerts: {len(alert_results['info'])}")
        print("="*60)
    
    def save_results(self, 
                    check_results: Dict[str, Any], 
                    alert_results: Dict[str, List[str]], 
                    output_file: str = "threshold_check_results.json"):
        """
        Save threshold check results to JSON file.
        
        Args:
            check_results: Results from threshold checking
            alert_results: Results from alert condition checking
            output_file: Output file path
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "threshold_check": check_results,
            "alerts": alert_results,
            "overall_status": "PASSED" if check_results["passed"] else "FAILED"
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")

def main():
    """Main function to run threshold checking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check evaluation results against thresholds")
    parser.add_argument("--metrics", "-m", default="results/metrics.json", 
                       help="Path to metrics.json file")
    parser.add_argument("--thresholds", "-t", default="thresholds.json", 
                       help="Path to thresholds.json file")
    parser.add_argument("--output", "-o", default="threshold_check_results.json", 
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Load metrics
    metrics_file = Path(args.metrics)
    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        sys.exit(1)
    
    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)
    
    # Initialize threshold checker
    checker = ThresholdChecker(args.thresholds)
    
    # Check summary metrics
    if "summary_metrics" not in metrics_data:
        print("❌ No summary metrics found in metrics file")
        sys.exit(1)
    
    summary_metrics = metrics_data["summary_metrics"]
    check_results = checker.check_summary_metrics(summary_metrics)
    alert_results = checker.check_alert_conditions(summary_metrics)
    
    # Print results
    checker.print_results(check_results, alert_results)
    
    # Save results
    checker.save_results(check_results, alert_results, args.output)
    
    # Exit with appropriate code
    if not check_results["passed"] or alert_results["critical"]:
        print("\n❌ THRESHOLD CHECK FAILED - Test suite should fail")
        sys.exit(1)
    else:
        print("\n✅ THRESHOLD CHECK PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main() 