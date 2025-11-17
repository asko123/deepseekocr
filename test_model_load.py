#!/usr/bin/env python3
"""
Test script to verify DeepSeek OCR model loading.
Run this after downloading model files to verify setup.
"""

import sys
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

def test_model_loading(model_path='./models/deepseek-ocr'):
    """Test loading the DeepSeek OCR model."""
    
    print("=" * 60)
    print("DeepSeek OCR Model Loading Test")
    print("=" * 60)
    
    # Check if model directory exists
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"❌ Error: Model directory not found: {model_path}")
        print(f"\nPlease follow MANUAL_SETUP.md to download model files.")
        sys.exit(1)
    
    print(f"✓ Model directory found: {model_path}")
    
    # Check for required files
    required_files = {
        'config.json': 'Model configuration',
        'modeling_deepseekocr.py': 'Custom model code',
        'tokenizer_config.json': 'Tokenizer configuration',
        'preprocessor_config.json': 'Preprocessor configuration'
    }
    missing_files = []
    
    for file, description in required_files.items():
        file_path = model_dir / file
        if file_path.exists():
            print(f"✓ Found: {file} ({description})")
        else:
            print(f"❌ Missing: {file} ({description})")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing required files. Please download from HuggingFace.")
        sys.exit(1)
    
    # Check for safetensors weight files only
    weight_files = list(model_dir.glob("*.safetensors"))
    if weight_files:
        print(f"✓ Found {len(weight_files)} safetensors weight file(s)")
        total_size = 0
        for wf in weight_files:
            size_gb = wf.stat().st_size / (1024**3)
            total_size += size_gb
            print(f"  - {wf.name} ({size_gb:.2f} GB)")
        print(f"  Total size: {total_size:.2f} GB")
    else:
        print("❌ No safetensors weight files found")
        print("   Please download model-*.safetensors files from HuggingFace")
        print("   Visit: https://huggingface.co/deepseek-ai/DeepSeek-OCR/tree/main")
        sys.exit(1)
    
    print("\n" + "-" * 60)
    print("Loading tokenizer...")
    print("-" * 60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {str(e)}")
        sys.exit(1)
    
    print("\n" + "-" * 60)
    print("Loading model (this may take a minute)...")
    print("-" * 60)
    
    try:
        # Try with flash attention first
        try:
            model = AutoModel.from_pretrained(
                model_path,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True,
                use_safetensors=True
            )
            print("✓ Model loaded with flash attention")
        except Exception as flash_err:
            print(f"⚠ Flash attention failed: {flash_err}")
            print("  Falling back to standard attention...")
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_safetensors=True
            )
            print("✓ Model loaded with standard attention")
        
        print(f"  Model type: {type(model).__name__}")
        print(f"  Model class: {model.__class__.__module__}.{model.__class__.__name__}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        sys.exit(1)
    
    print("\n" + "-" * 60)
    print("Moving model to GPU...")
    print("-" * 60)
    
    try:
        if not torch.cuda.is_available():
            print("⚠ CUDA not available, using CPU (this will be slow)")
            model = model.eval()
        else:
            model = model.eval().cuda().to(torch.bfloat16)
            print("✓ Model moved to GPU")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    except Exception as e:
        print(f"❌ Failed to move model to GPU: {str(e)}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now run the pipeline:")
    print(f"  python deepseek_ocr_pipeline.py sample.jpg --model-path {model_path}")
    print("\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DeepSeek OCR model loading')
    parser.add_argument(
        '--model-path',
        type=str,
        default='./models/deepseek-ocr',
        help='Path to model directory (default: ./models/deepseek-ocr)'
    )
    
    args = parser.parse_args()
    test_model_loading(args.model_path)

