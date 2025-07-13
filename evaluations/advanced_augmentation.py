"""
Advanced Data Augmentation System for Computer Vision

This module implements sophisticated data augmentation techniques specifically designed
for blink detection evaluation. It demonstrates advanced CV preprocessing capabilities
suitable for QA lead positions, including lighting variations, motion artifacts,
and realistic edge cases.
"""

import cv2
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import time
from scipy import ndimage


def gaussian(M, std):
    """Simple 1D Gaussian kernel (numpy fallback)."""
    n = np.arange(0, M) - (M - 1.0) / 2.0
    g = np.exp(-0.5 * (n / std) ** 2)
    return g / g.sum()


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    lighting_variations: bool = True
    motion_artifacts: bool = True
    noise_injection: bool = True
    compression_artifacts: bool = True
    resolution_variations: bool = True
    temporal_variations: bool = True
    edge_case_simulation: bool = True
    
    # Intensity parameters
    lighting_intensity: float = 0.3
    motion_intensity: float = 0.2
    noise_intensity: float = 0.1
    compression_quality: int = 70
    resolution_scale_range: Tuple[float, float] = (0.8, 1.2)
    temporal_shift_range: Tuple[int, int] = (-5, 5)


@dataclass
class AugmentationResult:
    """Result of data augmentation."""
    original_frame: np.ndarray
    augmented_frame: np.ndarray
    augmentation_type: str
    parameters: Dict[str, Any]
    quality_metrics: Dict[str, float]


class AdvancedAugmenter:
    """Advanced data augmentation system for CV evaluation."""
    
    def __init__(self, config: AugmentationConfig = None):
        """
        Initialize the advanced augmenter.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()
        self.augmentation_history = []
    
    def augment_video_sequence(self, frames: List[np.ndarray], 
                              labels: List[Dict] = None) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Apply advanced augmentation to a video sequence.
        
        Args:
            frames: List of video frames
            labels: Optional list of frame labels
            
        Returns:
            Tuple of (augmented_frames, augmented_labels)
        """
        augmented_frames = []
        augmented_labels = []
        
        for i, frame in enumerate(frames):
            # Apply multiple augmentation techniques
            augmented_frame = frame.copy()
            augmentation_params = {}
            
            # Lighting variations
            if self.config.lighting_variations:
                augmented_frame, lighting_params = self._apply_lighting_variations(augmented_frame)
                augmentation_params.update(lighting_params)
            
            # Motion artifacts
            if self.config.motion_artifacts:
                augmented_frame, motion_params = self._apply_motion_artifacts(augmented_frame)
                augmentation_params.update(motion_params)
            
            # Noise injection
            if self.config.noise_injection:
                augmented_frame, noise_params = self._apply_noise_injection(augmented_frame)
                augmentation_params.update(noise_params)
            
            # Compression artifacts
            if self.config.compression_artifacts:
                augmented_frame, compression_params = self._apply_compression_artifacts(augmented_frame)
                augmentation_params.update(compression_params)
            
            # Resolution variations
            if self.config.resolution_variations:
                augmented_frame, resolution_params = self._apply_resolution_variations(augmented_frame)
                augmentation_params.update(resolution_params)
            
            # Edge case simulation
            if self.config.edge_case_simulation and random.random() < 0.1:
                augmented_frame, edge_params = self._simulate_edge_cases(augmented_frame)
                augmentation_params.update(edge_params)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(frame, augmented_frame)
            
            # Store augmentation result
            result = AugmentationResult(
                original_frame=frame,
                augmented_frame=augmented_frame,
                augmentation_type="comprehensive",
                parameters=augmentation_params,
                quality_metrics=quality_metrics
            )
            self.augmentation_history.append(result)
            
            augmented_frames.append(augmented_frame)
            
            # Update labels if provided
            if labels and i < len(labels):
                augmented_label = labels[i].copy()
                augmented_label['augmentation_applied'] = True
                augmented_label['augmentation_params'] = augmentation_params
                augmented_labels.append(augmented_label)
            else:
                augmented_labels.append({
                    'augmentation_applied': True,
                    'augmentation_params': augmentation_params
                })
        
        return augmented_frames, augmented_labels
    
    def _apply_lighting_variations(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply realistic lighting variations."""
        params = {}
        
        # Random brightness adjustment
        brightness_factor = 1.0 + random.uniform(-self.config.lighting_intensity, 
                                               self.config.lighting_intensity)
        frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)
        params['brightness_factor'] = brightness_factor
        
        # Random contrast adjustment
        contrast_factor = 1.0 + random.uniform(-0.2, 0.2)
        frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=0)
        params['contrast_factor'] = contrast_factor
        
        # Simulate different lighting conditions
        lighting_type = random.choice(['natural', 'artificial', 'mixed'])
        if lighting_type == 'artificial':
            # Add warm/cool tint
            tint_factor = random.uniform(0.8, 1.2)
            frame = self._apply_color_tint(frame, tint_factor)
            params['lighting_type'] = lighting_type
            params['tint_factor'] = tint_factor
        
        return frame, params
    
    def _apply_motion_artifacts(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply realistic motion artifacts."""
        params = {}
        
        # Motion blur
        if random.random() < 0.3:
            kernel_size = random.randint(3, 7)
            angle = random.uniform(0, 360)
            frame = self._apply_motion_blur(frame, kernel_size, angle)
            params['motion_blur'] = {'kernel_size': kernel_size, 'angle': angle}
        
        # Camera shake
        if random.random() < 0.2:
            shake_intensity = random.uniform(1, 3)
            frame = self._apply_camera_shake(frame, shake_intensity)
            params['camera_shake'] = {'intensity': shake_intensity}
        
        return frame, params
    
    def _apply_noise_injection(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply realistic noise injection."""
        params = {}
        
        # Gaussian noise
        if random.random() < 0.4:
            noise_std = random.uniform(0, self.config.noise_intensity * 255)
            noise = np.random.normal(0, noise_std, frame.shape).astype(np.uint8)
            frame = cv2.add(frame, noise)
            params['gaussian_noise'] = {'std': noise_std}
        
        # Salt and pepper noise
        if random.random() < 0.2:
            salt_prob = random.uniform(0, 0.01)
            pepper_prob = random.uniform(0, 0.01)
            frame = self._apply_salt_pepper_noise(frame, salt_prob, pepper_prob)
            params['salt_pepper_noise'] = {'salt_prob': salt_prob, 'pepper_prob': pepper_prob}
        
        return frame, params
    
    def _apply_compression_artifacts(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply compression artifacts."""
        params = {}
        
        # JPEG compression simulation
        quality = random.randint(self.config.compression_quality, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        
        # Encode and decode to simulate compression
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        params['compression_quality'] = quality
        
        return frame, params
    
    def _apply_resolution_variations(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply resolution variations."""
        params = {}
        
        # Random scaling
        scale_factor = random.uniform(*self.config.resolution_scale_range)
        height, width = frame.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Resize back to original size
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        
        params['resolution_scale'] = scale_factor
        
        return frame, params
    
    def _simulate_edge_cases(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simulate realistic edge cases."""
        params = {}
        edge_case_type = random.choice([
            'partial_occlusion', 'extreme_lighting', 'rapid_motion', 'low_quality'
        ])
        
        if edge_case_type == 'partial_occlusion':
            frame = self._simulate_partial_occlusion(frame)
            params['edge_case'] = 'partial_occlusion'
        
        elif edge_case_type == 'extreme_lighting':
            frame = self._simulate_extreme_lighting(frame)
            params['edge_case'] = 'extreme_lighting'
        
        elif edge_case_type == 'rapid_motion':
            frame = self._simulate_rapid_motion(frame)
            params['edge_case'] = 'rapid_motion'
        
        elif edge_case_type == 'low_quality':
            frame = self._simulate_low_quality(frame)
            params['edge_case'] = 'low_quality'
        
        return frame, params
    
    def _apply_color_tint(self, frame: np.ndarray, tint_factor: float) -> np.ndarray:
        """Apply color tint to simulate different lighting."""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Adjust hue channel
        hsv[:, :, 0] = (hsv[:, :, 0] * tint_factor) % 180
        
        # Convert back to BGR
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _apply_motion_blur(self, frame: np.ndarray, kernel_size: int, angle: float) -> np.ndarray:
        """Apply motion blur with specified kernel size and angle."""
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1
        kernel = kernel / kernel_size
        
        # Rotate kernel
        center = (kernel_size // 2, kernel_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        
        # Apply blur
        return cv2.filter2D(frame, -1, kernel)
    
    def _apply_camera_shake(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Apply camera shake effect."""
        height, width = frame.shape[:2]
        
        # Create displacement field
        dx = np.random.normal(0, intensity, (height, width))
        dy = np.random.normal(0, intensity, (height, width))
        
        # Apply displacement
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        x_coords = np.clip(x_coords + dx, 0, width - 1).astype(np.float32)
        y_coords = np.clip(y_coords + dy, 0, height - 1).astype(np.float32)
        
        return cv2.remap(frame, x_coords, y_coords, cv2.INTER_LINEAR)
    
    def _apply_salt_pepper_noise(self, frame: np.ndarray, salt_prob: float, 
                                 pepper_prob: float) -> np.ndarray:
        """Apply salt and pepper noise."""
        noisy = frame.copy()
        
        # Salt noise
        salt_mask = np.random.random(frame.shape[:2]) < salt_prob
        noisy[salt_mask] = 255
        
        # Pepper noise
        pepper_mask = np.random.random(frame.shape[:2]) < pepper_prob
        noisy[pepper_mask] = 0
        
        return noisy
    
    def _simulate_partial_occlusion(self, frame: np.ndarray) -> np.ndarray:
        """Simulate partial occlusion."""
        height, width = frame.shape[:2]
        
        # Create random occlusion rectangle
        x1 = random.randint(0, width // 4)
        y1 = random.randint(0, height // 4)
        x2 = random.randint(x1 + width // 8, width // 2)
        y2 = random.randint(y1 + height // 8, height // 2)
        
        # Apply occlusion (darken the region)
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] // 2
        
        return frame
    
    def _simulate_extreme_lighting(self, frame: np.ndarray) -> np.ndarray:
        """Simulate extreme lighting conditions."""
        # Randomly choose between overexposure and underexposure
        if random.random() < 0.5:
            # Overexposure
            frame = cv2.convertScaleAbs(frame, alpha=2.0, beta=50)
        else:
            # Underexposure
            frame = cv2.convertScaleAbs(frame, alpha=0.3, beta=-50)
        
        return frame
    
    def _simulate_rapid_motion(self, frame: np.ndarray) -> np.ndarray:
        """Simulate rapid motion blur."""
        kernel_size = random.randint(9, 15)
        angle = random.uniform(0, 360)
        return self._apply_motion_blur(frame, kernel_size, angle)
    
    def _simulate_low_quality(self, frame: np.ndarray) -> np.ndarray:
        """Simulate low quality video."""
        # Downsample and upsample
        height, width = frame.shape[:2]
        small_frame = cv2.resize(frame, (width // 4, height // 4))
        frame = cv2.resize(small_frame, (width, height))
        
        # Add noise
        noise = np.random.normal(0, 20, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        return frame
    
    def _calculate_quality_metrics(self, original: np.ndarray, 
                                  augmented: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics between original and augmented frames."""
        # PSNR (Peak Signal-to-Noise Ratio)
        mse = np.mean((original.astype(np.float64) - augmented.astype(np.float64)) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # SSIM (Structural Similarity Index)
        ssim = self._calculate_ssim(original, augmented)
        
        # Mean absolute difference
        mad = np.mean(np.abs(original.astype(np.float64) - augmented.astype(np.float64)))
        
        return {
            'psnr': psnr,
            'ssim': ssim,
            'mean_absolute_difference': mad
        }
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM between two images."""
        # Simplified SSIM calculation
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.std(img1)
        sigma2 = np.std(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))
        
        return ssim


class AugmentationAnalyzer:
    """Analyzer for augmentation results and quality assessment."""
    
    def __init__(self, output_dir: str = "evaluations/results/augmentation_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_augmentation_quality(self, results: List[AugmentationResult]) -> Dict[str, Any]:
        """Analyze the quality of augmentation results."""
        analysis = {
            'total_frames': len(results),
            'quality_metrics': {},
            'augmentation_types': {},
            'edge_cases': {},
            'recommendations': []
        }
        
        # Aggregate quality metrics
        psnr_values = [r.quality_metrics.get('psnr', 0) for r in results]
        ssim_values = [r.quality_metrics.get('ssim', 0) for r in results]
        mad_values = [r.quality_metrics.get('mean_absolute_difference', 0) for r in results]
        
        analysis['quality_metrics'] = {
            'psnr': {
                'mean': np.mean(psnr_values),
                'std': np.std(psnr_values),
                'min': np.min(psnr_values),
                'max': np.max(psnr_values)
            },
            'ssim': {
                'mean': np.mean(ssim_values),
                'std': np.std(ssim_values),
                'min': np.min(ssim_values),
                'max': np.max(ssim_values)
            },
            'mean_absolute_difference': {
                'mean': np.mean(mad_values),
                'std': np.std(mad_values),
                'min': np.min(mad_values),
                'max': np.max(mad_values)
            }
        }
        
        # Analyze augmentation types
        augmentation_types = {}
        for result in results:
            aug_type = result.augmentation_type
            if aug_type not in augmentation_types:
                augmentation_types[aug_type] = 0
            augmentation_types[aug_type] += 1
        
        analysis['augmentation_types'] = augmentation_types
        
        # Analyze edge cases
        edge_cases = {}
        for result in results:
            edge_case = result.parameters.get('edge_case')
            if edge_case:
                if edge_case not in edge_cases:
                    edge_cases[edge_case] = 0
                edge_cases[edge_case] += 1
        
        analysis['edge_cases'] = edge_cases
        
        # Generate recommendations
        if np.mean(psnr_values) < 30:
            analysis['recommendations'].append("Low PSNR detected. Consider reducing augmentation intensity.")
        
        if np.mean(ssim_values) < 0.8:
            analysis['recommendations'].append("Low SSIM detected. Augmentation may be too aggressive.")
        
        if len(edge_cases) < len(results) * 0.1:
            analysis['recommendations'].append("Consider increasing edge case simulation frequency.")
        
        return analysis
    
    def create_quality_report(self, results: List[AugmentationResult]) -> str:
        """Create a comprehensive quality report."""
        analysis = self.analyze_augmentation_quality(results)
        
        # Create visualizations
        self._create_quality_visualizations(results, analysis)
        
        # Generate report
        report = f"""
        # Augmentation Quality Report
        
        ## Summary
        - Total frames processed: {analysis['total_frames']}
        - Average PSNR: {analysis['quality_metrics']['psnr']['mean']:.2f}
        - Average SSIM: {analysis['quality_metrics']['ssim']['mean']:.3f}
        - Average MAD: {analysis['quality_metrics']['mean_absolute_difference']['mean']:.2f}
        
        ## Quality Metrics
        - PSNR Range: {analysis['quality_metrics']['psnr']['min']:.2f} - {analysis['quality_metrics']['psnr']['max']:.2f}
        - SSIM Range: {analysis['quality_metrics']['ssim']['min']:.3f} - {analysis['quality_metrics']['ssim']['max']:.3f}
        
        ## Augmentation Types
        """
        
        for aug_type, count in analysis['augmentation_types'].items():
            report += f"- {aug_type}: {count} frames\n"
        
        report += "\n## Edge Cases\n"
        for edge_case, count in analysis['edge_cases'].items():
            report += f"- {edge_case}: {count} frames\n"
        
        report += "\n## Recommendations\n"
        for rec in analysis['recommendations']:
            report += f"- {rec}\n"
        
        # Save report
        report_path = self.output_dir / 'quality_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        return str(report_path)
    
    def _create_quality_visualizations(self, results: List[AugmentationResult], 
                                     analysis: Dict[str, Any]):
        """Create quality visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Quality metrics distribution
        psnr_values = [r.quality_metrics.get('psnr', 0) for r in results]
        ssim_values = [r.quality_metrics.get('ssim', 0) for r in results]
        
        axes[0, 0].hist(psnr_values, bins=20, alpha=0.7, label='PSNR')
        axes[0, 0].set_title('PSNR Distribution')
        axes[0, 0].set_xlabel('PSNR')
        axes[0, 0].set_ylabel('Frequency')
        
        # Plot 2: SSIM distribution
        axes[0, 1].hist(ssim_values, bins=20, alpha=0.7, color='orange')
        axes[0, 1].set_title('SSIM Distribution')
        axes[0, 1].set_xlabel('SSIM')
        axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Augmentation types
        aug_types = list(analysis['augmentation_types'].keys())
        aug_counts = list(analysis['augmentation_types'].values())
        axes[1, 0].bar(aug_types, aug_counts)
        axes[1, 0].set_title('Augmentation Types')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Edge cases
        edge_cases = list(analysis['edge_cases'].keys())
        edge_counts = list(analysis['edge_cases'].values())
        if edge_cases:
            axes[1, 1].pie(edge_counts, labels=edge_cases, autopct='%1.1f%%')
            axes[1, 1].set_title('Edge Case Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'No edge cases', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_advanced_augmentation_pipeline(config: AugmentationConfig = None) -> AdvancedAugmenter:
    """
    Create an advanced augmentation pipeline.
    
    Args:
        config: Optional augmentation configuration
        
    Returns:
        AdvancedAugmenter instance
    """
    return AdvancedAugmenter(config)


def augment_evaluation_dataset(frames: List[np.ndarray], 
                             labels: List[Dict] = None,
                             config: AugmentationConfig = None) -> Tuple[List[np.ndarray], List[Dict], Dict[str, Any]]:
    """
    Apply advanced augmentation to an evaluation dataset.
    
    Args:
        frames: List of video frames
        labels: Optional list of frame labels
        config: Optional augmentation configuration
        
    Returns:
        Tuple of (augmented_frames, augmented_labels, quality_analysis)
    """
    augmenter = AdvancedAugmenter(config)
    analyzer = AugmentationAnalyzer()
    
    # Apply augmentation
    augmented_frames, augmented_labels = augmenter.augment_video_sequence(frames, labels)
    
    # Analyze quality
    quality_analysis = analyzer.analyze_augmentation_quality(augmenter.augmentation_history)
    
    # Create quality report
    report_path = analyzer.create_quality_report(augmenter.augmentation_history)
    
    return augmented_frames, augmented_labels, quality_analysis 