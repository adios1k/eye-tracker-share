#!/usr/bin/env python3
"""
Script to download videos from Google Drive for CI environments.
"""

import os
import sys
import subprocess
from pathlib import Path

def install_gdown():
    """Install gdown if not available."""
    try:
        import gdown
        print("✅ gdown already available")
    except ImportError:
        print("📦 Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

def download_from_drive():
    """Download videos from Google Drive."""
    # Install gdown
    install_gdown()
    
    # Import gdown after installation
    import gdown
    
    # Create videos directory
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)
    
    # Google Drive file IDs from your shared links
    # These are the file IDs extracted from your Google Drive URLs
    drive_files = {
        "video_1.mp4": "1OA8che-4PNjZrgr_8vUP4UcLZ4BC7UFv",
        "video_2.mp4": "1zdAvOOdJ0zld_J0Zi9YIP9-Zyb0FzD5C",
        "video_3.mp4": "1BSiC0_dQljXy863cH-8_ngWF6khsarjX",
        "video_4.mp4": "1hB_NVS9Jg4iTHn6WqpwL8erlR_9sZ6zT",
        "video_5.mp4": "1jAEk_gCt3Co3T6y5xiO_T4Nf7McK9BPX",
        "video_6.mp4": "111ZkYjLBKshdanPS8f1Bq0DVaWJDBRau",
        "video_7.mp4": "1leTUr-eVr63x0spfqiQNMnbXZu8Rr5z3",
        "video_8.mp4": "1hAPdqQpO9HhfgpzHX6U0D89JVWIKw58e",
        "video_9.mp4": "1PXHsXfxP177vsoT-jyv_H8MIsPTnUrPT",
        "video_10.mp4": "1PJsRlCI9wFii7JUus7EOvZTl0mhbymA4",
        "video_11.mp4": "1dCR8aHeyBrgmGfpRvqq0lfW5ePeeT-Pw",
        "video_12.mp4": "1SsuvBr0-O3u9Gaym0xA9jXMUCFamNPXK",
        "video_13.mp4": "1z2GO9UwlHoGZlk2B5lrH68haHLgEwoxB",
        "video_14.mp4": "12HoJK8O1X9TkDFBmTWT-rVWyiEc4EXrL"
    }
    
    print("📥 Downloading videos from Google Drive...")
    
    for filename, file_id in drive_files.items():
        if file_id == "YOUR_VIDEO_FILE_ID_HERE" or file_id == "YOUR_JSON_FILE_ID_HERE":
            print(f"⚠️  Please update the file ID for {filename} in download_videos.py")
            continue
            
        output_path = videos_dir / filename
        
        if output_path.exists():
            print(f"✅ {filename} already exists, skipping...")
            continue
            
        try:
            print(f"📥 Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(output_path), quiet=False)
            
            if output_path.exists():
                print(f"✅ Successfully downloaded {filename}")
            else:
                print(f"❌ Failed to download {filename}")
                
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
    
    # Move files to evaluations directory for compatibility
    evaluations_dir = Path("evaluations")
    evaluations_dir.mkdir(exist_ok=True)
    
    for file_path in videos_dir.glob("*"):
        target_path = evaluations_dir / file_path.name
        if not target_path.exists():
            file_path.rename(target_path)
            print(f"📁 Moved {file_path.name} to evaluations directory")

if __name__ == "__main__":
    download_from_drive() 