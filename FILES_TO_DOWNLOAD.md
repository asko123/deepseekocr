# Files to Download from HuggingFace

Visit: https://huggingface.co/deepseek-ai/DeepSeek-OCR/tree/main

## Required Files (Download these to `models/deepseek-ocr/`)

### Configuration Files (Small, text files)
1. ✅ `config.json` - Model architecture configuration
2. ✅ `modeling_deepseekocr.py` - Custom DeepSeek OCR implementation
3. ✅ `tokenizer_config.json` - Tokenizer settings
4. ✅ `preprocessor_config.json` - Image preprocessing configuration
5. ✅ `generation_config.json` - Text generation parameters

### Model Weight Files (Large, safetensors format ONLY)

**Important Notes:**
- ❌ DO NOT download `.bin` files (not supported)
- ✅ ONLY download `.safetensors` files
- ❌ DO NOT download Flax/JAX files (PyTorch only)

**Check the repository for actual filenames. Could be:**

**Option A: Single file**
- `model.safetensors` (~20GB)

**Option B: Sharded files** (more common)
- `model-00001-of-00004.safetensors`
- `model-00002-of-00004.safetensors`
- `model-00003-of-00004.safetensors`
- `model-00004-of-00004.safetensors`

(Check the Files tab on HuggingFace for actual shard count)

## Files You DON'T Need

❌ `tokenizer.model` - NOT used by DeepSeek-OCR
❌ `vocab.json` - NOT used by DeepSeek-OCR  
❌ `merges.txt` - NOT used by DeepSeek-OCR
❌ `*.bin` files - Use safetensors instead
❌ `flax_model.msgpack` - PyTorch only, not Flax
❌ `tf_model.h5` - PyTorch only, not TensorFlow

## Quick Download Commands

```bash
cd models/deepseek-ocr

# Download config files (small)
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/config.json
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/modeling_deepseekocr.py
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/tokenizer_config.json
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/preprocessor_config.json
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/generation_config.json

# Check the Files tab to see if it's a single file or sharded
# Then download the appropriate safetensors files

# For single file:
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model.safetensors

# OR for sharded (example with 4 shards):
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00001-of-00004.safetensors
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00002-of-00004.safetensors
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00003-of-00004.safetensors
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00004-of-00004.safetensors
```

## Verification

After downloading, verify:

```bash
cd models/deepseek-ocr

# Check config files
ls -lh *.json *.py

# Check safetensors files (should be several GB each)
ls -lh *.safetensors
du -sh *.safetensors

# Run test
cd ../..
python test_model_load.py
```

Expected output from test script:
```
✓ Model directory found
✓ Found: config.json (Model configuration)
✓ Found: modeling_deepseekocr.py (Custom model code)
✓ Found: tokenizer_config.json (Tokenizer configuration)
✓ Found: preprocessor_config.json (Preprocessor configuration)
✓ Found X safetensors weight file(s)
  Total size: XX.XX GB
✓ Tokenizer loaded successfully
✓ Model loaded
✓ Model moved to GPU
✓ ALL TESTS PASSED!
```

## Alternative: Use Git LFS (Easiest)

```bash
cd models/deepseek-ocr
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR .
```

This will automatically download all necessary files.

## Alternative: Use HuggingFace CLI

```bash
pip install --user huggingface-hub
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir ./models/deepseek-ocr
```

## Troubleshooting

**Q: I see both .bin and .safetensors files, which do I use?**
A: Download ONLY .safetensors files. The code is configured for safetensors format.

**Q: Where is tokenizer.model?**
A: DeepSeek-OCR doesn't use tokenizer.model. It uses transformers' built-in tokenizer with tokenizer_config.json.

**Q: How do I know if the model is sharded?**
A: Check the Files tab on HuggingFace. If you see files like `model-00001-of-00004.safetensors`, it's sharded. Download all shards.

**Q: Total file size?**
A: Expect 15-25GB total for the safetensors files, plus a few KB for config files.

