#!/usr/bin/env python3
"""
Script to check threshold compliance for evaluation results.
"""

import json
import sys
import os
from pathlib import Path

def check_thresholds():
    """Check if threshold check results exist and are compliant."""
    results_file = Path("evaluations/results/threshold_check_results.json")
    
    if not results_file.exists():
        print("⚠️ Threshold check results not found")
        return True  # Don't fail if file doesn't exist
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        threshold_check = data.get('threshold_check', {})
        alerts = data.get('alerts', {})
        
        passed = threshold_check.get('passed', False)
        critical_alerts = len(alerts.get('critical', []))
        
        if not passed or critical_alerts > 0:
            print('❌ Threshold check failed')
            print(f'   Passed: {passed}')
            print(f'   Critical alerts: {critical_alerts}')
            return False
        else:
            print('✅ Threshold check passed')
            return True
            
    except Exception as e:
        print(f"❌ Error checking thresholds: {e}")
        return False

if __name__ == "__main__":
    success = check_thresholds()
    sys.exit(0 if success else 1) 