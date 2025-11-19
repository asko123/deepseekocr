#!/usr/bin/env python3
"""
Download DeepSeek OCR model from HuggingFace.
This script downloads all necessary files (code, configs, and weights).
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Install it with: pip install --user huggingface-hub")
    sys.exit(1)


def download_deepseek_ocr(local_dir='./deepseek-ocr', resume=True):
    """
    Download DeepSeek OCR model from HuggingFace.
    
    Args:
        local_dir: Local directory to save the model
        resume: Resume interrupted downloads
    """
    model_id = "deepseek-ai/DeepSeek-OCR"  # Fixed typo: deekseek -> deepseek
    
    print("=" * 60)
    print("DeepSeek OCR Model Download")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Download location: {local_dir}")
    print()
    
    # Create directory if it doesn't exist
    local_path = Path(local_dir)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check available disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(local_path.parent)
        free_gb = free / (1024**3)
        print(f"Available disk space: {free_gb:.2f} GB")
        
        if free_gb < 30:
            print("⚠ Warning: Less than 30GB free. Model is ~20-25GB.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Download cancelled.")
                sys.exit(0)
    except:
        pass
    
    print()
    print("Starting download... (this may take 20-60 minutes)")
    print("You can safely interrupt and resume later with this same script.")
    print()
    
    try:
        # Download all files
        snapshot_download(
            model_id,
            local_dir=local_dir,
            resume_download=resume,
            local_dir_use_symlinks=False  # Use actual files, not symlinks
        )
        
        print()
        print("=" * 60)
        print("✓ Download complete!")
        print("=" * 60)
        print()
        
        # Verify downloaded files
        print("Verifying files...")
        
        required_files = [
            'config.json',
            'modeling_deepseekocr.py',
            'tokenizer_config.json',
            'preprocessor_config.json'
        ]
        
        missing_files = []
        for file in required_files:
            file_path = local_path / file
            if file_path.exists():
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} - MISSING")
                missing_files.append(file)
        
        # Check for safetensors files
        safetensors = list(local_path.glob("*.safetensors"))
        if safetensors:
            print(f"  ✓ Found {len(safetensors)} weight file(s)")
            total_size = sum(f.stat().st_size for f in safetensors) / (1024**3)
            print(f"    Total size: {total_size:.2f} GB")
        else:
            print("  ✗ No .safetensors weight files found")
            missing_files.append("*.safetensors")
        
        print()
        
        if missing_files:
            print("⚠ Warning: Some files are missing:")
            for f in missing_files:
                print(f"  - {f}")
            print()
            print("Try running the script again to resume the download.")
            sys.exit(1)
        else:
            print("✓ All required files present!")
            print()
            print("Next steps:")
            print(f"  1. Verify setup: python test_model_load.py --model-path {local_dir}")
            print(f"  2. Run pipeline: python deepseek_ocr_pipeline.py input.jpg --model-path {local_dir} --no-flash-attn")
            print()
        
    except KeyboardInterrupt:
        print()
        print("Download interrupted. Run this script again to resume.")
        sys.exit(0)
    except Exception as e:
        print()
        print(f"✗ Download failed: {str(e)}")
        print()
        print("Possible solutions:")
        print("  1. Check your internet connection")
        print("  2. Make sure you have enough disk space (~30GB)")
        print("  3. Try running the script again (downloads can be resumed)")
        print("  4. If you see 'rate limit' errors, wait a few hours and try again")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download DeepSeek OCR model from HuggingFace'
    )
    parser.add_argument(
        '--local-dir',
        type=str,
        default='./deepseek-ocr',
        help='Local directory to save the model (default: ./deepseek-ocr)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume interrupted downloads (start fresh)'
    )
    
    args = parser.parse_args()
    
    download_deepseek_ocr(
        local_dir=args.local_dir,
        resume=not args.no_resume
    )

