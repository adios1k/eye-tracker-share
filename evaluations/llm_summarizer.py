"""
LLM-Powered Evaluation Summary System

This module provides intelligent analysis and summarization of CV evaluation results
using LLM integration. It demonstrates advanced AI capabilities for QA lead positions
by providing contextual insights, trend analysis, and actionable recommendations.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import openai
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


@dataclass
class SummaryInsight:
    """Represents a specific insight from the evaluation."""
    category: str
    title: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float
    actionable: bool
    recommendations: List[str]


@dataclass
class EvaluationSummary:
    """Comprehensive evaluation summary with LLM-generated insights."""
    overall_score: float
    key_metrics: Dict[str, float]
    insights: List[SummaryInsight]
    trends: Dict[str, Any]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]
    performance_analysis: Dict[str, Any]


class LLMSummarizer:
    """LLM-powered summarization system for CV evaluation results."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the LLM summarizer.
        
        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            model: LLM model to use for summarization
        """
        self.model = model
        if api_key:
            openai.api_key = api_key
        else:
            # Try to get from environment
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                openai.api_key = api_key
            else:
                print("Warning: No OpenAI API key found. Using mock summarization.")
                self.model = "mock"
    
    def generate_comprehensive_summary(self, 
                                    metrics: Dict[str, Any],
                                    historical_data: Optional[List[Dict]] = None) -> EvaluationSummary:
        """
        Generate a comprehensive evaluation summary using LLM analysis.
        
        Args:
            metrics: Current evaluation metrics
            historical_data: Optional historical evaluation data for trend analysis
            
        Returns:
            EvaluationSummary object with LLM-generated insights
        """
        # Prepare data for LLM analysis
        analysis_data = self._prepare_analysis_data(metrics, historical_data)
        
        # Generate LLM analysis
        if self.model == "mock":
            llm_analysis = self._generate_mock_analysis(analysis_data)
        else:
            llm_analysis = self._generate_llm_analysis(analysis_data)
        
        # Parse and structure the analysis
        summary = self._parse_llm_analysis(llm_analysis, metrics)
        
        return summary
    
    def _prepare_analysis_data(self, metrics: Dict[str, Any], 
                              historical_data: Optional[List[Dict]]) -> Dict[str, Any]:
        """Prepare data for LLM analysis."""
        analysis_data = {
            'current_metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'evaluation_context': {
                'model_type': 'blink_detection',
                'evaluation_framework': 'comprehensive_cv_evaluation',
                'metrics_categories': list(metrics.keys())
            }
        }
        
        if historical_data:
            analysis_data['historical_trends'] = self._extract_trends(historical_data)
        
        return analysis_data
    
    def _generate_llm_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Generate analysis using OpenAI LLM."""
        prompt = self._create_analysis_prompt(analysis_data)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self._generate_mock_analysis(analysis_data)
    
    def _generate_mock_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Generate mock analysis when LLM is not available."""
        metrics = analysis_data['current_metrics']
        
        # Extract key metrics
        accuracy = metrics.get('accuracy', 0.0)
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        f1_score = metrics.get('f1_score', 0.0)
        
        # Generate insights based on metrics
        insights = []
        
        if accuracy < 0.8:
            insights.append("MODEL_PERFORMANCE: The model shows suboptimal accuracy. Consider retraining with more diverse data.")
        
        if precision < 0.7:
            insights.append("PRECISION_ISSUE: High false positive rate detected. Model may be over-predicting blinks.")
        
        if recall < 0.7:
            insights.append("RECALL_ISSUE: Low recall indicates missed blinks. Consider adjusting detection sensitivity.")
        
        if f1_score < 0.75:
            insights.append("BALANCE_ISSUE: F1 score indicates poor balance between precision and recall.")
        
        # Add temporal consistency insights
        temporal_metrics = metrics.get('temporal_consistency', {})
        if temporal_metrics:
            stability = temporal_metrics.get('prediction_stability', 0.0)
            if stability < 0.6:
                insights.append("TEMPORAL_INSTABILITY: Model predictions show high temporal instability.")
        
        # Add confidence calibration insights
        calibration_metrics = metrics.get('confidence_calibration', {})
        if calibration_metrics:
            calibration_error = calibration_metrics.get('calibration_error', 1.0)
            if calibration_error > 0.1:
                insights.append("CALIBRATION_ISSUE: Model confidence is poorly calibrated.")
        
        # Add edge case insights
        edge_case_metrics = metrics.get('edge_case_analysis', {})
        if edge_case_metrics:
            robustness = edge_case_metrics.get('robustness_score', 0.0)
            if robustness < 0.7:
                insights.append("ROBUSTNESS_ISSUE: Model shows vulnerability to edge cases.")
        
        # Generate recommendations
        recommendations = [
            "Consider data augmentation to improve model robustness",
            "Implement confidence threshold tuning",
            "Add more diverse training data",
            "Consider ensemble methods for improved stability"
        ]
        
        # Calculate overall score
        overall_score = (accuracy + precision + recall + f1_score) / 4
        
        analysis = f"""
        OVERALL_SCORE: {overall_score:.3f}
        
        KEY_INSIGHTS:
        {chr(10).join(insights)}
        
        RECOMMENDATIONS:
        {chr(10).join(recommendations)}
        
        RISK_ASSESSMENT:
        - Performance Risk: {'HIGH' if overall_score < 0.7 else 'MEDIUM' if overall_score < 0.8 else 'LOW'}
        - Deployment Readiness: {'NOT READY' if overall_score < 0.75 else 'READY WITH MONITORING' if overall_score < 0.85 else 'READY'}
        
        TREND_ANALYSIS:
        - Model shows {'improving' if overall_score > 0.8 else 'stable' if overall_score > 0.7 else 'declining'} performance trend
        """
        
        return analysis
    
    def _create_analysis_prompt(self, analysis_data: Dict[str, Any]) -> str:
        """Create analysis prompt for LLM."""
        metrics = analysis_data['current_metrics']
        
        prompt = f"""
        Analyze the following computer vision evaluation metrics for a blink detection model:

        EVALUATION METRICS:
        {json.dumps(metrics, indent=2)}

        CONTEXT:
        - Model Type: Blink Detection
        - Evaluation Framework: Comprehensive CV Evaluation
        - Timestamp: {analysis_data['timestamp']}

        Please provide a comprehensive analysis including:
        1. Overall performance assessment with numerical score (0-1)
        2. Key insights categorized by severity (low/medium/high/critical)
        3. Specific recommendations for improvement
        4. Risk assessment for deployment
        5. Trend analysis if historical data is available
        6. Performance profiling insights

        Format your response with clear sections and actionable insights.
        """
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """
        You are an expert computer vision QA analyst specializing in model evaluation and performance assessment. 
        Your role is to analyze evaluation metrics and provide actionable insights for model improvement.
        
        Key responsibilities:
        1. Identify performance bottlenecks and areas for improvement
        2. Assess model robustness and reliability
        3. Provide specific, actionable recommendations
        4. Evaluate deployment readiness
        5. Analyze trends and patterns in model performance
        
        Always be specific, data-driven, and actionable in your analysis.
        """
    
    def _parse_llm_analysis(self, analysis: str, metrics: Dict[str, Any]) -> EvaluationSummary:
        """Parse LLM analysis into structured summary."""
        # Extract overall score
        overall_score_match = re.search(r'OVERALL_SCORE:\s*([\d.]+)', analysis)
        overall_score = float(overall_score_match.group(1)) if overall_score_match else 0.5
        
        # Extract insights
        insights = self._extract_insights(analysis)
        
        # Extract recommendations
        recommendations = self._extract_recommendations(analysis)
        
        # Extract risk assessment
        risk_assessment = self._extract_risk_assessment(analysis)
        
        # Calculate key metrics
        key_metrics = self._calculate_key_metrics(metrics)
        
        # Generate trends
        trends = self._generate_trends(metrics)
        
        # Generate performance analysis
        performance_analysis = self._generate_performance_analysis(metrics)
        
        return EvaluationSummary(
            overall_score=overall_score,
            key_metrics=key_metrics,
            insights=insights,
            trends=trends,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            performance_analysis=performance_analysis
        )
    
    def _extract_insights(self, analysis: str) -> List[SummaryInsight]:
        """Extract insights from LLM analysis."""
        insights = []
        
        # Parse insights section
        insights_section = re.search(r'KEY_INSIGHTS:(.*?)(?=RECOMMENDATIONS:|$)', 
                                   analysis, re.DOTALL)
        if insights_section:
            insight_lines = insights_section.group(1).strip().split('\n')
            for line in insight_lines:
                if line.strip():
                    # Parse insight format: "CATEGORY: description"
                    match = re.match(r'(\w+):\s*(.+)', line.strip())
                    if match:
                        category, description = match.groups()
                        insight = SummaryInsight(
                            category=category,
                            title=f"{category.replace('_', ' ').title()}",
                            description=description,
                            severity=self._determine_severity(description),
                            confidence=0.8,  # Default confidence
                            actionable=True,
                            recommendations=[description]
                        )
                        insights.append(insight)
        
        return insights
    
    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract recommendations from LLM analysis."""
        recommendations = []
        
        # Parse recommendations section
        rec_section = re.search(r'RECOMMENDATIONS:(.*?)(?=RISK_ASSESSMENT:|$)', 
                               analysis, re.DOTALL)
        if rec_section:
            rec_lines = rec_section.group(1).strip().split('\n')
            for line in rec_lines:
                if line.strip() and not line.startswith('-'):
                    recommendations.append(line.strip())
        
        return recommendations
    
    def _extract_risk_assessment(self, analysis: str) -> Dict[str, Any]:
        """Extract risk assessment from LLM analysis."""
        risk_assessment = {}
        
        # Parse risk assessment section
        risk_section = re.search(r'RISK_ASSESSMENT:(.*?)(?=TREND_ANALYSIS:|$)', 
                                analysis, re.DOTALL)
        if risk_section:
            risk_lines = risk_section.group(1).strip().split('\n')
            for line in risk_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    risk_assessment[key.strip()] = value.strip()
        
        return risk_assessment
    
    def _calculate_key_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate key metrics from evaluation results."""
        key_metrics = {}
        
        # Basic metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in metrics:
                key_metrics[metric] = metrics[metric]
        
        # Advanced metrics
        if 'temporal_consistency' in metrics:
            temporal = metrics['temporal_consistency']
            key_metrics['temporal_stability'] = temporal.get('prediction_stability', 0.0)
            key_metrics['temporal_coherence'] = temporal.get('temporal_coherence', 0.0)
        
        if 'confidence_calibration' in metrics:
            calibration = metrics['confidence_calibration']
            key_metrics['calibration_error'] = calibration.get('calibration_error', 1.0)
        
        if 'edge_case_analysis' in metrics:
            edge_cases = metrics['edge_case_analysis']
            key_metrics['robustness_score'] = edge_cases.get('robustness_score', 0.0)
        
        return key_metrics
    
    def _generate_trends(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trend analysis."""
        trends = {
            'performance_trend': 'stable',
            'improvement_areas': [],
            'regression_areas': []
        }
        
        # Analyze performance trends
        if 'accuracy' in metrics:
            accuracy = metrics['accuracy']
            if accuracy < 0.7:
                trends['performance_trend'] = 'declining'
                trends['regression_areas'].append('accuracy')
            elif accuracy > 0.9:
                trends['performance_trend'] = 'improving'
                trends['improvement_areas'].append('accuracy')
        
        return trends
    
    def _generate_performance_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance analysis."""
        analysis = {
            'strengths': [],
            'weaknesses': [],
            'optimization_opportunities': []
        }
        
        # Analyze strengths and weaknesses
        if 'accuracy' in metrics:
            if metrics['accuracy'] > 0.85:
                analysis['strengths'].append('High accuracy')
            elif metrics['accuracy'] < 0.7:
                analysis['weaknesses'].append('Low accuracy')
        
        if 'precision' in metrics:
            if metrics['precision'] < 0.7:
                analysis['weaknesses'].append('High false positive rate')
                analysis['optimization_opportunities'].append('Tune precision threshold')
        
        if 'recall' in metrics:
            if metrics['recall'] < 0.7:
                analysis['weaknesses'].append('Low recall - missed detections')
                analysis['optimization_opportunities'].append('Improve sensitivity')
        
        return analysis
    
    def _determine_severity(self, description: str) -> str:
        """Determine severity level from description."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['critical', 'severe', 'broken']):
            return 'critical'
        elif any(word in description_lower for word in ['high', 'poor', 'bad']):
            return 'high'
        elif any(word in description_lower for word in ['medium', 'moderate']):
            return 'medium'
        else:
            return 'low'
    
    def _extract_trends(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Extract trends from historical data."""
        if not historical_data:
            return {}
        
        # Calculate trends over time
        trends = {
            'accuracy_trend': [],
            'precision_trend': [],
            'recall_trend': [],
            'f1_trend': []
        }
        
        for data_point in historical_data:
            if 'accuracy' in data_point:
                trends['accuracy_trend'].append(data_point['accuracy'])
            if 'precision' in data_point:
                trends['precision_trend'].append(data_point['precision'])
            if 'recall' in data_point:
                trends['recall_trend'].append(data_point['recall'])
            if 'f1_score' in data_point:
                trends['f1_trend'].append(data_point['f1_score'])
        
        return trends


class SummaryVisualizer:
    """Visualization utilities for LLM-generated summaries."""
    
    def __init__(self, output_dir: str = "evaluations/results/summaries"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_summary_dashboard(self, summary: EvaluationSummary, 
                               metrics: Dict[str, Any]) -> str:
        """Create a comprehensive summary dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Overall score
        axes[0, 0].pie([summary.overall_score, 1 - summary.overall_score], 
                       labels=['Performance', 'Gap'], autopct='%1.1f%%')
        axes[0, 0].set_title('Overall Performance Score')
        
        # Plot 2: Key metrics radar chart
        self._plot_radar_chart(axes[0, 1], summary.key_metrics)
        
        # Plot 3: Insights by severity
        self._plot_insights_by_severity(axes[0, 2], summary.insights)
        
        # Plot 4: Risk assessment
        self._plot_risk_assessment(axes[1, 0], summary.risk_assessment)
        
        # Plot 5: Performance analysis
        self._plot_performance_analysis(axes[1, 1], summary.performance_analysis)
        
        # Plot 6: Trends
        self._plot_trends(axes[1, 2], summary.trends)
        
        plt.tight_layout()
        dashboard_path = self.output_dir / 'summary_dashboard.png'
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(dashboard_path)
    
    def _plot_radar_chart(self, ax, key_metrics: Dict[str, float]):
        """Plot radar chart of key metrics."""
        if not key_metrics:
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center')
            return
        
        categories = list(key_metrics.keys())
        values = list(key_metrics.values())
        
        # Normalize values to 0-1 range
        values = [max(0, min(1, v)) for v in values]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Key Metrics Overview')
    
    def _plot_insights_by_severity(self, ax, insights: List[SummaryInsight]):
        """Plot insights grouped by severity."""
        severity_counts = {}
        for insight in insights:
            severity = insight.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if severity_counts:
            severities = list(severity_counts.keys())
            counts = list(severity_counts.values())
            colors = ['green', 'yellow', 'orange', 'red']
            
            ax.bar(severities, counts, color=colors[:len(severities)])
            ax.set_title('Insights by Severity')
            ax.set_ylabel('Count')
        else:
            ax.text(0.5, 0.5, 'No insights available', ha='center', va='center')
    
    def _plot_risk_assessment(self, ax, risk_assessment: Dict[str, Any]):
        """Plot risk assessment."""
        if not risk_assessment:
            ax.text(0.5, 0.5, 'No risk assessment available', ha='center', va='center')
            return
        
        risks = list(risk_assessment.keys())
        levels = list(risk_assessment.values())
        
        # Color code risk levels
        colors = []
        for level in levels:
            if 'HIGH' in level.upper():
                colors.append('red')
            elif 'MEDIUM' in level.upper():
                colors.append('orange')
            else:
                colors.append('green')
        
        ax.bar(risks, [1] * len(risks), color=colors)
        ax.set_title('Risk Assessment')
        ax.set_xticklabels(risks, rotation=45)
    
    def _plot_performance_analysis(self, ax, performance_analysis: Dict[str, Any]):
        """Plot performance analysis."""
        categories = ['Strengths', 'Weaknesses', 'Opportunities']
        counts = [
            len(performance_analysis.get('strengths', [])),
            len(performance_analysis.get('weaknesses', [])),
            len(performance_analysis.get('optimization_opportunities', []))
        ]
        
        colors = ['green', 'red', 'blue']
        ax.bar(categories, counts, color=colors)
        ax.set_title('Performance Analysis')
        ax.set_ylabel('Count')
    
    def _plot_trends(self, ax, trends: Dict[str, Any]):
        """Plot trend analysis."""
        if not trends:
            ax.text(0.5, 0.5, 'No trend data available', ha='center', va='center')
            return
        
        trend_text = f"Performance Trend: {trends.get('performance_trend', 'Unknown')}"
        ax.text(0.5, 0.5, trend_text, ha='center', va='center', fontsize=12)
        ax.set_title('Trend Analysis')
        ax.axis('off')


def generate_llm_summary(metrics: Dict[str, Any], 
                        historical_data: Optional[List[Dict]] = None,
                        api_key: Optional[str] = None) -> Tuple[EvaluationSummary, str]:
    """
    Generate LLM-powered summary of evaluation results.
    
    Args:
        metrics: Evaluation metrics
        historical_data: Optional historical data for trend analysis
        api_key: Optional OpenAI API key
        
    Returns:
        Tuple of (EvaluationSummary, dashboard_path)
    """
    summarizer = LLMSummarizer(api_key=api_key)
    visualizer = SummaryVisualizer()
    
    # Generate summary
    summary = summarizer.generate_comprehensive_summary(metrics, historical_data)
    
    # Create dashboard
    dashboard_path = visualizer.create_summary_dashboard(summary, metrics)
    
    return summary, dashboard_path 