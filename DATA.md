# Dataset Documentation

## Overview
This repository contains augmented video data for testing blink detection model robustness under various conditions.

## Dataset Structure

### Original Data
- **Video**: `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25.mp4`
- **Labels**: `Labels_Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25.json`

### Augmented Videos (Local)
The following augmented videos are available locally in `evaluations/augmented_videos/`:

#### Lighting Variations
- `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25_bright_dark.mp4` (8.7MB)
- `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25_low_contrast.mp4` (13MB)
- `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25_high_contrast.mp4` (47MB)

#### Quality Variations
- `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25_blur_light.mp4` (22MB)
- `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25_blur_heavy.mp4` (17MB)
- `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25_low_saturation.mp4` (29MB)
- `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25_high_saturation.mp4` (32MB)

#### Compression Variations
- `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25_compression_low.mp4` (35MB)
- `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25_compression_medium.mp4` (39MB)

#### Scale Variations
- `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25_scale_small.mp4` (258B)
- `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25_scale_large.mp4` (258B)

### Large Files (Google Drive)
The following large files (>50MB) are stored in Google Drive due to GitHub file size limits:

#### High-Intensity Noise Variations
- `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25_noise_heavy.mp4` (542MB)
- `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25_noise_light.mp4` (541MB)

#### Bright Lighting Variation
- `Mehul blink recording_ nightlight_specs and nospecs_ 02_03_25_bright_light.mp4` (55MB)

**Google Drive Link**: [Eye Tracker Augmented Videos](https://drive.google.com/drive/folders/1Y8o7lAuLzfzojoZQAXuccEgXH62fckA-?usp=drive_link)


## Ground Truth Labels
Each augmented video has a corresponding JSON label file with the same name (e.g., `video_name.json`). The labels contain frame-by-frame annotations with:
- `open_closed`: "Open" or "Closed" eye state
- `direction`: Gaze direction ("Up", "Down", "Straight", etc.)

## Augmentation Types

### 1. Lighting Variations
- **Bright Dark**: Reduced brightness (factor: 0.3)
- **Bright Light**: Increased brightness (factor: 1.8)
- **Low Contrast**: Reduced contrast (factor: 0.5)
- **High Contrast**: Increased contrast (factor: 1.5)

### 2. Quality Variations
- **Light Blur**: Gaussian blur with kernel size 3
- **Heavy Blur**: Gaussian blur with kernel size 9
- **Low Saturation**: Reduced saturation (factor: 0.3)
- **High Saturation**: Increased saturation (factor: 1.8)

### 3. Noise Variations
- **Light Noise**: Gaussian noise with intensity 0.05
- **Heavy Noise**: Gaussian noise with intensity 0.2

### 4. Compression Variations
- **Low Compression**: JPEG quality 30
- **Medium Compression**: JPEG quality 60

### 5. Scale Variations
- **Small Scale**: 50% of original size
- **Large Scale**: 150% of original size

## Usage

### Download Large Files
1. Access the Google Drive folder using the link above
2. Download the required large video files
3. Place them in `evaluations/augmented_videos/` directory

### Running Evaluations
```bash
# The evaluation scripts will automatically use all available videos
python evaluations/run_evaluation.py
```

## Metadata
The `augmentation_metadata.json` file contains:
- Original video path
- List of all generated augmentations
- Total count of generated videos

## Notes
- Large files (>50MB) are excluded from git to avoid GitHub size limits
- All label files are included in the repository
- The augmentation script can regenerate any missing videos if needed 