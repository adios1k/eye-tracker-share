#!/usr/bin/env python3
"""
Script to download videos from Google Drive using GitLab CI/CD variables.
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
    """Download videos from Google Drive using GitLab variables."""
    # Install gdown
    install_gdown()
    
    # Import gdown after installation
    import gdown
    
    # Get file IDs from GitLab CI/CD variables
    # You can set these as comma-separated values in GitLab CI/CD variables
    video_file_ids = os.environ.get("GDRIVE_VIDEO_FILE_IDS", "").split(",")
    
    if not video_file_ids or video_file_ids[0] == "":
        print("⚠️  GitLab CI/CD variable not set:")
        print("   - GDRIVE_VIDEO_FILE_IDS (comma-separated list of file IDs)")
        print("   Please set this in GitLab CI/CD Settings > Variables")
        print("   Example: 1OA8che-4PNjZrgr_8vUP4UcLZ4BC7UFv,1zdAvOOdJ0zld_J0Zi9YIP9-Zyb0FzD5C,...")
        return
    
    # Create videos directory
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)
    
    # Define files to download
    drive_files = {}
    for i, file_id in enumerate(video_file_ids, 1):
        file_id = file_id.strip()
        if file_id:
            drive_files[f"video_{i}.mp4"] = file_id
    
    print("📥 Downloading videos from Google Drive...")
    
    for filename, file_id in drive_files.items():
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
    
    # Move files to root directory for compatibility
    for file_path in videos_dir.glob("*"):
        target_path = Path(file_path.name)
        if not target_path.exists():
            file_path.rename(target_path)
            print(f"📁 Moved {file_path.name} to root directory")

if __name__ == "__main__":
    download_from_drive() 