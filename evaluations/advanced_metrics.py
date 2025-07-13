"""
Advanced Computer Vision Evaluation Metrics

This module implements sophisticated metrics for evaluating blink detection models,
including temporal consistency analysis, confidence calibration, edge case detection,
and performance profiling. These metrics demonstrate advanced CV evaluation capabilities
suitable for a QA lead position.
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
try:
    from sklearn.metrics import calibration_curve
except ImportError:
    # Fallback for older sklearn versions
    def calibration_curve(y_true, y_score, n_bins=10):
        """Simple calibration curve implementation for older sklearn versions."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        fraction_of_positives = []
        mean_predicted_value = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = np.logical_and(y_score > bin_lower, y_score <= bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                fraction_of_positives.append(np.mean(y_true[in_bin]))
                mean_predicted_value.append(np.mean(y_score[in_bin]))
            else:
                fraction_of_positives.append(0)
                mean_predicted_value.append((bin_lower + bin_upper) / 2)
        
        return np.array(fraction_of_positives), np.array(mean_predicted_value)
import cv2
from pathlib import Path


@dataclass
class TemporalConsistencyMetrics:
    """Metrics for analyzing temporal consistency of predictions."""
    prediction_stability: float  # How stable predictions are over time
    transition_smoothness: float  # Smoothness of state transitions
    temporal_coherence: float  # Overall temporal coherence score
    false_oscillation_rate: float  # Rate of rapid state changes
    mean_prediction_duration: float  # Average duration of predictions


@dataclass
class ConfidenceCalibrationMetrics:
    """Metrics for analyzing confidence calibration."""
    calibration_error: float  # Expected calibration error
    reliability_diagram: Dict[str, List[float]]  # Reliability diagram data
    confidence_histogram: Dict[str, List[float]]  # Confidence distribution
    overconfidence_score: float  # Measure of overconfidence
    underconfidence_score: float  # Measure of underconfidence


@dataclass
class EdgeCaseMetrics:
    """Metrics for detecting and analyzing edge cases."""
    edge_case_count: int  # Number of detected edge cases
    edge_case_types: Dict[str, int]  # Types of edge cases found
    edge_case_severity: Dict[str, float]  # Severity scores for edge cases
    robustness_score: float  # Overall robustness to edge cases


@dataclass
class PerformanceProfile:
    """Detailed performance profiling metrics."""
    inference_time_stats: Dict[str, float]  # Inference time statistics
    memory_usage_stats: Dict[str, float]  # Memory usage statistics
    throughput_metrics: Dict[str, float]  # Throughput metrics
    resource_efficiency: float  # Overall resource efficiency score


class AdvancedMetricsCalculator:
    """Advanced metrics calculator for comprehensive CV evaluation."""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_temporal_consistency(self, predictions: List[Dict], 
                                     ground_truth: List[Dict]) -> TemporalConsistencyMetrics:
        """
        Calculate temporal consistency metrics for predictions.
        
        Args:
            predictions: List of prediction dictionaries with timestamps
            ground_truth: List of ground truth dictionaries with timestamps
            
        Returns:
            TemporalConsistencyMetrics object
        """
        # Extract prediction sequences
        pred_sequence = [p.get('blink_detected', False) for p in predictions]
        gt_sequence = [gt.get('blink_detected', False) for gt in ground_truth]
        
        # Calculate prediction stability (variance of prediction durations)
        prediction_durations = self._calculate_prediction_durations(pred_sequence)
        prediction_stability = 1.0 / (1.0 + np.var(prediction_durations)) if prediction_durations else 0.0
        
        # Calculate transition smoothness
        transitions = self._count_transitions(pred_sequence)
        transition_smoothness = 1.0 / (1.0 + transitions)
        
        # Calculate temporal coherence (correlation with ground truth)
        # Convert boolean sequences to numeric for correlation
        pred_numeric = np.array([1 if p else 0 for p in pred_sequence])
        gt_numeric = np.array([1 if g else 0 for g in gt_sequence])
        
        # Ensure arrays are the same length
        min_length = min(len(pred_numeric), len(gt_numeric))
        if min_length > 1:
            pred_numeric = pred_numeric[:min_length]
            gt_numeric = gt_numeric[:min_length]
            try:
                correlation_matrix = np.corrcoef(pred_numeric, gt_numeric)
                temporal_coherence = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
                temporal_coherence = max(0, temporal_coherence)  # Ensure non-negative
            except (ValueError, IndexError, TypeError):
                temporal_coherence = 0.0
        else:
            temporal_coherence = 0.0
        
        # Calculate false oscillation rate
        false_oscillations = self._count_false_oscillations(pred_sequence, gt_sequence)
        false_oscillation_rate = false_oscillations / len(pred_sequence) if pred_sequence else 0.0
        
        # Calculate mean prediction duration
        mean_prediction_duration = np.mean(prediction_durations) if prediction_durations else 0.0
        
        return TemporalConsistencyMetrics(
            prediction_stability=float(prediction_stability),
            transition_smoothness=float(transition_smoothness),
            temporal_coherence=float(temporal_coherence),
            false_oscillation_rate=float(false_oscillation_rate),
            mean_prediction_duration=float(mean_prediction_duration)
        )
    
    def calculate_confidence_calibration(self, predictions: List[Dict], 
                                       ground_truth: List[Dict]) -> ConfidenceCalibrationMetrics:
        """
        Calculate confidence calibration metrics.
        
        Args:
            predictions: List of prediction dictionaries with confidence scores
            ground_truth: List of ground truth dictionaries
            
        Returns:
            ConfidenceCalibrationMetrics object
        """
        # Extract confidence scores and ground truth
        confidences = [p.get('confidence', 0.5) for p in predictions]
        gt_labels = [gt.get('blink_detected', False) for gt in ground_truth]
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            gt_labels, confidences, n_bins=10
        )
        
        # Calculate expected calibration error
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Calculate over/under confidence scores
        overconfidence_score = self._calculate_overconfidence(confidences, gt_labels)
        underconfidence_score = self._calculate_underconfidence(confidences, gt_labels)
        
        # Create reliability diagram data
        reliability_diagram = {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist()
        }
        
        # Create confidence histogram
        confidence_histogram = {
            'bins': np.linspace(0, 1, 20).tolist(),
            'counts': np.histogram(confidences, bins=20)[0].tolist()
        }
        
        return ConfidenceCalibrationMetrics(
            calibration_error=float(calibration_error),
            reliability_diagram=reliability_diagram,
            confidence_histogram=confidence_histogram,
            overconfidence_score=float(overconfidence_score),
            underconfidence_score=float(underconfidence_score)
        )
    
    def detect_edge_cases(self, predictions: List[Dict], 
                         ground_truth: List[Dict],
                         video_metadata: Dict) -> EdgeCaseMetrics:
        """
        Detect and analyze edge cases in predictions.
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries
            video_metadata: Video metadata including lighting, motion, etc.
            
        Returns:
            EdgeCaseMetrics object
        """
        edge_cases = []
        edge_case_types = defaultdict(int)
        edge_case_severity = {}
        
        # Detect rapid state changes (potential false positives)
        rapid_changes = self._detect_rapid_changes(predictions)
        edge_cases.extend(rapid_changes)
        edge_case_types['rapid_state_changes'] = len(rapid_changes)
        
        # Detect confidence anomalies
        confidence_anomalies = self._detect_confidence_anomalies(predictions)
        edge_cases.extend(confidence_anomalies)
        edge_case_types['confidence_anomalies'] = len(confidence_anomalies)
        
        # Detect lighting-related issues
        lighting_issues = self._detect_lighting_issues(predictions, video_metadata)
        edge_cases.extend(lighting_issues)
        edge_case_types['lighting_issues'] = len(lighting_issues)
        
        # Detect motion-related issues
        motion_issues = self._detect_motion_issues(predictions, video_metadata)
        edge_cases.extend(motion_issues)
        edge_case_types['motion_issues'] = len(motion_issues)
        
        # Calculate severity scores
        for edge_case in edge_cases:
            severity = self._calculate_edge_case_severity(edge_case)
            edge_case_severity[edge_case['id']] = severity
        
        # Calculate overall robustness score
        robustness_score = 1.0 - (len(edge_cases) / len(predictions))
        robustness_score = max(0, robustness_score)
        
        return EdgeCaseMetrics(
            edge_case_count=len(edge_cases),
            edge_case_types=dict(edge_case_types),
            edge_case_severity=edge_case_severity,
            robustness_score=robustness_score
        )
    
    def profile_performance(self, inference_times: List[float],
                           memory_usage: List[float],
                           frame_counts: List[int]) -> PerformanceProfile:
        """
        Profile performance metrics.
        
        Args:
            inference_times: List of inference times in seconds
            memory_usage: List of memory usage in MB
            frame_counts: List of frame counts processed
            
        Returns:
            PerformanceProfile object
        """
        # Calculate inference time statistics
        inference_time_stats = {
            'mean': float(np.mean(inference_times)),
            'std': float(np.std(inference_times)),
            'min': float(np.min(inference_times)),
            'max': float(np.max(inference_times)),
            'p95': float(np.percentile(inference_times, 95)),
            'p99': float(np.percentile(inference_times, 99))
        }
        
        # Calculate memory usage statistics
        memory_usage_stats = {
            'mean': float(np.mean(memory_usage)),
            'std': float(np.std(memory_usage)),
            'min': float(np.min(memory_usage)),
            'max': float(np.max(memory_usage)),
            'peak': float(np.max(memory_usage))
        }
        
        # Calculate throughput metrics
        total_frames = sum(frame_counts)
        total_time = sum(inference_times)
        throughput_metrics = {
            'frames_per_second': float(total_frames / total_time if total_time > 0 else 0),
            'average_latency': float(total_time / len(inference_times) if inference_times else 0),
            'efficiency': float(total_frames / (total_time * np.mean(memory_usage)) if total_time > 0 and np.mean(memory_usage) > 0 else 0)
        }
        
        # Calculate overall resource efficiency
        memory_mean = float(memory_usage_stats['mean'])
        fps = float(throughput_metrics['frames_per_second'])
        resource_efficiency = (fps * (1.0 / (1.0 + memory_mean / 1000)))
        
        return PerformanceProfile(
            inference_time_stats=inference_time_stats,
            memory_usage_stats=memory_usage_stats,
            throughput_metrics=throughput_metrics,
            resource_efficiency=resource_efficiency
        )
    
    def _calculate_prediction_durations(self, sequence: List[bool]) -> List[int]:
        """Calculate durations of consecutive predictions."""
        durations = []
        current_duration = 1
        current_state = sequence[0] if sequence else False
        
        for i in range(1, len(sequence)):
            if sequence[i] == current_state:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_duration = 1
                current_state = sequence[i]
        
        durations.append(current_duration)
        return durations
    
    def _count_transitions(self, sequence: List[bool]) -> int:
        """Count the number of state transitions."""
        transitions = 0
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                transitions += 1
        return transitions
    
    def _count_false_oscillations(self, predictions: List[bool], 
                                 ground_truth: List[bool]) -> int:
        """Count false oscillations (rapid changes not in ground truth)."""
        false_oscillations = 0
        for i in range(1, len(predictions)):
            if (predictions[i] != predictions[i-1] and 
                ground_truth[i] == ground_truth[i-1]):
                false_oscillations += 1
        return false_oscillations
    
    def _calculate_overconfidence(self, confidences: List[float], 
                                labels: List[bool]) -> float:
        """Calculate overconfidence score."""
        overconfident_predictions = []
        for conf, label in zip(confidences, labels):
            if conf > 0.8 and not label:  # High confidence but wrong
                overconfident_predictions.append(conf)
        return np.mean(overconfident_predictions) if overconfident_predictions else 0
    
    def _calculate_underconfidence(self, confidences: List[float], 
                                  labels: List[bool]) -> float:
        """Calculate underconfidence score."""
        underconfident_predictions = []
        for conf, label in zip(confidences, labels):
            if conf < 0.3 and label:  # Low confidence but correct
                underconfident_predictions.append(1 - conf)
        return np.mean(underconfident_predictions) if underconfident_predictions else 0
    
    def _detect_rapid_changes(self, predictions: List[Dict]) -> List[Dict]:
        """Detect rapid state changes in predictions."""
        rapid_changes = []
        for i in range(1, len(predictions)):
            if (predictions[i].get('blink_detected') != 
                predictions[i-1].get('blink_detected')):
                # Check if this is a rapid change
                time_diff = predictions[i].get('timestamp', 0) - predictions[i-1].get('timestamp', 0)
                if time_diff < 0.1:  # Less than 100ms
                    rapid_changes.append({
                        'id': f'rapid_change_{i}',
                        'type': 'rapid_state_change',
                        'timestamp': predictions[i].get('timestamp', 0),
                        'details': f'State change in {time_diff:.3f}s'
                    })
        return rapid_changes
    
    def _detect_confidence_anomalies(self, predictions: List[Dict]) -> List[Dict]:
        """Detect confidence anomalies."""
        anomalies = []
        confidences = [p.get('confidence', 0.5) for p in predictions]
        mean_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        for i, pred in enumerate(predictions):
            confidence = pred.get('confidence', 0.5)
            if abs(confidence - mean_confidence) > 2 * std_confidence:
                anomalies.append({
                    'id': f'confidence_anomaly_{i}',
                    'type': 'confidence_anomaly',
                    'timestamp': pred.get('timestamp', 0),
                    'details': f'Confidence {confidence:.3f} vs mean {mean_confidence:.3f}'
                })
        return anomalies
    
    def _detect_lighting_issues(self, predictions: List[Dict], 
                               video_metadata: Dict) -> List[Dict]:
        """Detect lighting-related issues."""
        issues = []
        # This would typically analyze video frames for lighting conditions
        # For now, we'll create a placeholder implementation
        return issues
    
    def _detect_motion_issues(self, predictions: List[Dict], 
                             video_metadata: Dict) -> List[Dict]:
        """Detect motion-related issues."""
        issues = []
        # This would typically analyze video frames for motion artifacts
        # For now, we'll create a placeholder implementation
        return issues
    
    def _calculate_edge_case_severity(self, edge_case: Dict) -> float:
        """Calculate severity score for an edge case."""
        # Simple severity calculation based on type
        severity_map = {
            'rapid_state_change': 0.8,
            'confidence_anomaly': 0.6,
            'lighting_issue': 0.7,
            'motion_issue': 0.5
        }
        return severity_map.get(edge_case['type'], 0.5)


class MetricsVisualizer:
    """Visualization utilities for advanced metrics."""
    
    def __init__(self, output_dir: str = "evaluations/results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_temporal_consistency(self, temporal_metrics: TemporalConsistencyMetrics,
                                 predictions: List[Dict], ground_truth: List[Dict]):
        """Create temporal consistency visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Prediction stability
        axes[0, 0].bar(['Stability', 'Smoothness', 'Coherence'], 
                       [temporal_metrics.prediction_stability,
                        temporal_metrics.transition_smoothness,
                        temporal_metrics.temporal_coherence])
        axes[0, 0].set_title('Temporal Consistency Metrics')
        axes[0, 0].set_ylim(0, 1)
        
        # Plot 2: Prediction sequence
        pred_sequence = [p.get('blink_detected', False) for p in predictions]
        gt_sequence = [gt.get('blink_detected', False) for gt in ground_truth]
        
        axes[0, 1].plot(pred_sequence, label='Predictions', alpha=0.7)
        axes[0, 1].plot(gt_sequence, label='Ground Truth', alpha=0.7)
        axes[0, 1].set_title('Prediction vs Ground Truth Sequence')
        axes[0, 1].legend()
        
        # Plot 3: False oscillation rate
        axes[1, 0].pie([temporal_metrics.false_oscillation_rate, 
                       1 - temporal_metrics.false_oscillation_rate],
                       labels=['False Oscillations', 'Stable Predictions'],
                       autopct='%1.1f%%')
        axes[1, 0].set_title('False Oscillation Analysis')
        
        # Plot 4: Prediction duration distribution
        prediction_durations = self._calculate_prediction_durations(pred_sequence)
        axes[1, 1].hist(prediction_durations, bins=20, alpha=0.7)
        axes[1, 1].set_title('Prediction Duration Distribution')
        axes[1, 1].set_xlabel('Duration (frames)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_consistency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confidence_calibration(self, calibration_metrics: ConfidenceCalibrationMetrics):
        """Create confidence calibration visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Reliability diagram
        reliability = calibration_metrics.reliability_diagram
        axes[0, 0].plot(reliability['mean_predicted_value'], 
                        reliability['fraction_of_positives'], 'o-')
        axes[0, 0].plot([0, 1], [0, 1], '--', color='gray')
        axes[0, 0].set_xlabel('Mean Predicted Value')
        axes[0, 0].set_ylabel('Fraction of Positives')
        axes[0, 0].set_title('Reliability Diagram')
        axes[0, 0].grid(True)
        
        # Plot 2: Confidence histogram
        histogram = calibration_metrics.confidence_histogram
        axes[0, 1].bar(histogram['bins'][:-1], histogram['counts'], 
                       width=0.05, alpha=0.7)
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Confidence Distribution')
        
        # Plot 3: Calibration error
        axes[1, 0].bar(['Calibration Error'], [calibration_metrics.calibration_error])
        axes[1, 0].set_title('Calibration Error')
        axes[1, 0].set_ylim(0, 1)
        
        # Plot 4: Over/Under confidence
        axes[1, 1].bar(['Overconfidence', 'Underconfidence'], 
                       [calibration_metrics.overconfidence_score,
                        calibration_metrics.underconfidence_score])
        axes[1, 1].set_title('Confidence Bias Analysis')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_calibration.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_edge_case_analysis(self, edge_case_metrics: EdgeCaseMetrics):
        """Create edge case analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Edge case types
        types = list(edge_case_metrics.edge_case_types.keys())
        counts = list(edge_case_metrics.edge_case_types.values())
        axes[0, 0].bar(types, counts)
        axes[0, 0].set_title('Edge Case Types')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Robustness score
        axes[0, 1].pie([edge_case_metrics.robustness_score, 
                       1 - edge_case_metrics.robustness_score],
                       labels=['Robust', 'Vulnerable'],
                       autopct='%1.1f%%')
        axes[0, 1].set_title('Robustness Analysis')
        
        # Plot 3: Edge case severity distribution
        severities = list(edge_case_metrics.edge_case_severity.values())
        if severities:
            axes[1, 0].hist(severities, bins=10, alpha=0.7)
            axes[1, 0].set_title('Edge Case Severity Distribution')
            axes[1, 0].set_xlabel('Severity Score')
            axes[1, 0].set_ylabel('Count')
        
        # Plot 4: Overall metrics
        metrics = ['Edge Cases', 'Robustness Score']
        values = [edge_case_metrics.edge_case_count, edge_case_metrics.robustness_score]
        axes[1, 1].bar(metrics, values)
        axes[1, 1].set_title('Overall Edge Case Metrics')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'edge_case_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_profile(self, performance_profile: PerformanceProfile):
        """Create performance profile visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Inference time distribution
        inference_stats = performance_profile.inference_time_stats
        axes[0, 0].bar(['Mean', 'Std', 'P95', 'P99'], 
                       [inference_stats['mean'], inference_stats['std'],
                        inference_stats['p95'], inference_stats['p99']])
        axes[0, 0].set_title('Inference Time Statistics')
        axes[0, 0].set_ylabel('Time (seconds)')
        
        # Plot 2: Memory usage
        memory_stats = performance_profile.memory_usage_stats
        axes[0, 1].bar(['Mean', 'Peak'], 
                       [memory_stats['mean'], memory_stats['peak']])
        axes[0, 1].set_title('Memory Usage Statistics')
        axes[0, 1].set_ylabel('Memory (MB)')
        
        # Plot 3: Throughput metrics
        throughput = performance_profile.throughput_metrics
        axes[1, 0].bar(['FPS', 'Latency', 'Efficiency'], 
                       [throughput['frames_per_second'], 
                        throughput['average_latency'],
                        throughput['efficiency']])
        axes[1, 0].set_title('Throughput Metrics')
        
        # Plot 4: Resource efficiency
        axes[1, 1].bar(['Resource Efficiency'], [performance_profile.resource_efficiency])
        axes[1, 1].set_title('Overall Resource Efficiency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_profile.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_prediction_durations(self, sequence: List[bool]) -> List[int]:
        """Calculate durations of consecutive predictions."""
        durations = []
        current_duration = 1
        current_state = sequence[0] if sequence else False
        
        for i in range(1, len(sequence)):
            if sequence[i] == current_state:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_duration = 1
                current_state = sequence[i]
        
        durations.append(current_duration)
        return durations


def calculate_advanced_metrics(predictions: List[Dict], 
                             ground_truth: List[Dict],
                             video_metadata: Dict = None,
                             inference_times: List[float] = None,
                             memory_usage: List[float] = None) -> Dict[str, Any]:
    """
    Calculate all advanced metrics for comprehensive CV evaluation.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        video_metadata: Optional video metadata
        inference_times: Optional list of inference times
        memory_usage: Optional list of memory usage
        
    Returns:
        Dictionary containing all advanced metrics
    """
    try:
        calculator = AdvancedMetricsCalculator()
        visualizer = MetricsVisualizer()
        
        # Calculate temporal consistency
        temporal_metrics = calculator.calculate_temporal_consistency(predictions, ground_truth)
        
        # Calculate confidence calibration
        calibration_metrics = calculator.calculate_confidence_calibration(predictions, ground_truth)
        
        # Detect edge cases
        edge_case_metrics = calculator.detect_edge_cases(predictions, ground_truth, video_metadata or {})
        
        # Profile performance
        performance_profile = None
        if inference_times and memory_usage:
            frame_counts = [1] * len(inference_times)  # Assuming 1 frame per inference
            performance_profile = calculator.profile_performance(inference_times, memory_usage, frame_counts)
        
        # Create visualizations
        try:
            visualizer.plot_temporal_consistency(temporal_metrics, predictions, ground_truth)
            visualizer.plot_confidence_calibration(calibration_metrics)
            visualizer.plot_edge_case_analysis(edge_case_metrics)
            if performance_profile:
                visualizer.plot_performance_profile(performance_profile)
        except Exception as viz_error:
            print(f"Warning: Visualization failed: {viz_error}")
        
        # Compile results
        results = {
            'temporal_consistency': {
                'prediction_stability': temporal_metrics.prediction_stability,
                'transition_smoothness': temporal_metrics.transition_smoothness,
                'temporal_coherence': temporal_metrics.temporal_coherence,
                'false_oscillation_rate': temporal_metrics.false_oscillation_rate,
                'mean_prediction_duration': temporal_metrics.mean_prediction_duration
            },
            'confidence_calibration': {
                'calibration_error': calibration_metrics.calibration_error,
                'overconfidence_score': calibration_metrics.overconfidence_score,
                'underconfidence_score': calibration_metrics.underconfidence_score
            },
            'edge_case_analysis': {
                'edge_case_count': edge_case_metrics.edge_case_count,
                'edge_case_types': edge_case_metrics.edge_case_types,
                'robustness_score': edge_case_metrics.robustness_score
            }
        }
        
        if performance_profile:
            results['performance_profile'] = {
                'inference_time_stats': performance_profile.inference_time_stats,
                'memory_usage_stats': performance_profile.memory_usage_stats,
                'throughput_metrics': performance_profile.throughput_metrics,
                'resource_efficiency': performance_profile.resource_efficiency
            }
        
        return results
        
    except Exception as e:
        print(f"Error in calculate_advanced_metrics: {e}")
        import traceback
        traceback.print_exc()
        
        # Return fallback results
        return {
            'temporal_consistency': {
                'prediction_stability': 0.0,
                'transition_smoothness': 0.0,
                'temporal_coherence': 0.0,
                'false_oscillation_rate': 0.0,
                'mean_prediction_duration': 0.0
            },
            'confidence_calibration': {
                'calibration_error': 0.0,
                'overconfidence_score': 0.0,
                'underconfidence_score': 0.0
            },
            'edge_case_analysis': {
                'edge_case_count': 0,
                'edge_case_types': {},
                'robustness_score': 0.0
            }
        } 