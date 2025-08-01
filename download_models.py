#!/usr/bin/env python3
"""
Model Download Script for Email NLP Classification API

This script downloads the required model files for the Email NLP Classification API.
The models are hosted on Google Drive for easy access.

Usage:
    python download_models.py
"""

import os
import requests
import zipfile
from pathlib import Path

def download_file(url, filename):
    """Download a file from URL with progress bar."""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    print(f"\nDownloaded {filename} successfully!")

def main():
    """Main function to download model files."""
    print("🚀 Email NLP Classification API - Model Downloader")
    print("=" * 50)
    
    # Create directories if they don't exist
    Path("models").mkdir(exist_ok=True)
    Path("production/models").mkdir(exist_ok=True)
    
    # Model file URLs (you'll need to upload these to a cloud service)
    model_files = {
        "models/perfect_accuracy_model_20250730_171812.pkl": "MODEL_URL_HERE",
        "models/perfect_accuracy_label_encoder_20250730_171812.pkl": "LABEL_ENCODER_URL_HERE", 
        "models/perfect_accuracy_scaler_20250730_171812.pkl": "SCALER_URL_HERE",
        "production/models/final_calibrated_model_20250730_150602.pkl": "PRODUCTION_MODEL_URL_HERE",
        "production/models/final_label_encoder_20250730_150602.pkl": "PRODUCTION_LABEL_ENCODER_URL_HERE",
        "production/models/final_scaler_20250730_150602.pkl": "PRODUCTION_SCALER_URL_HERE"
    }
    
    print("⚠️  Note: Model files are large (~700MB total)")
    print("📥 Downloading model files...")
    
    for filename, url in model_files.items():
        if url != "URL_HERE":  # Replace with actual URLs
            download_file(url, filename)
        else:
            print(f"⚠️  {filename} - URL not configured")
    
    print("\n✅ Model download complete!")
    print("📝 Next steps:")
    print("1. Install dependencies: pip install -r production/requirements.txt")
    print("2. Start the API: python production/production_api.py")
    print("3. Test the API: curl -X POST http://localhost:8080/classify -H 'Content-Type: application/json' -d '{\"email\": \"Hello, I need help with my computer\"}'")

if __name__ == "__main__":
    main() 