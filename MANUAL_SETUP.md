# Manual DeepSeek OCR Model Setup

This guide provides step-by-step instructions for manually downloading and setting up the DeepSeek OCR model weights.

## Prerequisites

- Python 3.12.9
- Git (for cloning repositories)
- At least 50GB free disk space for model files
- NVIDIA GPU with CUDA 11.8

## Step 1: Create Model Directory

```bash
cd /Users/Tawfiq/Desktop/deepseekocr
mkdir -p models/deepseek-ocr
cd models/deepseek-ocr
```

## Step 2: Download Model Files from HuggingFace

### Option A: Using Git LFS (Recommended)

```bash
# Install git-lfs if not already installed
# macOS: brew install git-lfs
# Linux: sudo apt-get install git-lfs

git lfs install

# Clone the model repository
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR .
```

### Option B: Manual Download (No Git LFS)

Visit: https://huggingface.co/deepseek-ai/DeepSeek-OCR

Download these files to `models/deepseek-ocr/`:

**Essential Files:**
1. `config.json` - Model configuration
2. `modeling_deepseekocr.py` - Custom model implementation  
3. `tokenizer_config.json` - Tokenizer configuration
4. `preprocessor_config.json` - Preprocessor configuration
5. Model weight files (safetensors format):
   - `model.safetensors` (single file), OR
   - `model-00001-of-xxxxx.safetensors` (sharded files)

**Additional Files:**
- `generation_config.json` - Generation parameters
- `README.md` - Model documentation
- `.gitattributes` - Git LFS configuration

**Note:** DeepSeek-OCR uses transformers' built-in tokenizer. You do NOT need separate `tokenizer.model` or `vocab.json` files.

### Download Commands (wget alternative)

```bash
cd models/deepseek-ocr

# Download core configuration files
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/config.json
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/modeling_deepseekocr.py
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/tokenizer_config.json
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/preprocessor_config.json
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/generation_config.json

# Download safetensors weight files (these are large - may take time)
# For single file model:
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model.safetensors

# OR for sharded models (check the Files tab for actual filenames):
# wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00001-of-00004.safetensors
# wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00002-of-00004.safetensors
# wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00003-of-00004.safetensors
# wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00004-of-00004.safetensors

# Note: Check https://huggingface.co/deepseek-ai/DeepSeek-OCR/tree/main 
# for the actual number of shard files
```

## Step 3: Verify Downloaded Files

```bash
cd models/deepseek-ocr
ls -lh

# Expected output should include:
# - config.json
# - modeling_deepseekocr.py
# - tokenizer files
# - model weight files (several GB in size)
```

Verify file integrity:

```bash
# Check that safetensors weight files are not empty
du -sh *.safetensors 2>/dev/null

# Should show files in GB range (e.g., 4.8G, 9.2G, etc.)

# Verify core config files exist
ls -lh config.json modeling_deepseekocr.py tokenizer_config.json preprocessor_config.json
```

## Step 4: Copy Custom Model Code

The `modeling_deepseekocr.py` file should already be in your model directory. Verify it contains the custom implementation:

```bash
grep -q "class DeepSeekOCR" models/deepseek-ocr/modeling_deepseekocr.py && echo "Model code found" || echo "Model code missing"
```

## Step 5: Update Pipeline Configuration

The pipeline code has been updated to support local model loading. Verify the configuration:

```python
# In deepseek_ocr_pipeline.py, the model is loaded from local path:
model_name = './models/deepseek-ocr'  # Local directory
```

## Step 6: Install Dependencies

```bash
# Install all required packages
pip install --user -r requirements.txt --no-build-isolation

# Verify transformers installation
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

## Step 7: Test Model Loading

Create a test script to verify the model loads correctly:

```python
# test_model_load.py
import torch
from transformers import AutoModel, AutoTokenizer

model_path = './models/deepseek-ocr'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

print("Loading model...")
model = AutoModel.from_pretrained(
    model_path,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)

print("Moving to GPU...")
model = model.eval().cuda().to(torch.bfloat16)

print("✓ Model loaded successfully!")
print(f"Model type: {type(model)}")
```

Run the test:

```bash
python test_model_load.py
```

## Step 8: Run the Pipeline

Test with a sample image:

```bash
python deepseek_ocr_pipeline.py sample_document.jpg -o output/
```

## Troubleshooting

### Issue: "No module named 'modeling_deepseekocr'"

**Solution:** Ensure `modeling_deepseekocr.py` is in the model directory and `trust_remote_code=True` is set.

### Issue: "Error loading model weights"

**Solution:** 
- Verify all weight files are downloaded completely
- Check file sizes match those on HuggingFace
- Try deleting and re-downloading corrupted files

### Issue: "CUDA out of memory"

**Solution:**
- Reduce batch size in processing
- Use a GPU with more VRAM
- Process smaller images

### Issue: "Unable to load safetensors"

**Solution:**
```bash
pip install --user safetensors>=0.4.0
```

### Issue: "Tokenizer files not found"

**Solution:** DeepSeek-OCR uses the transformers library's built-in tokenizer. You only need:
- `tokenizer_config.json` 
- `preprocessor_config.json`

You do NOT need `tokenizer.model` or `vocab.json` files.

### Issue: "Flash attention not found" or Installation Fails

**Solution:** Flash attention is optional. You have two options:

**Option 1: Skip flash-attn (Recommended)**
```bash
# Run pipeline without flash attention
python deepseek_ocr_pipeline.py document.pdf --no-flash-attn
```

**Option 2: Try installing (requires CUDA 11.8 + GPU)**
```bash
pip install --user flash-attn==2.7.3 --no-build-isolation
```

If installation fails with compilation errors, use Option 1. The model works fine without flash attention, just slightly slower.

## Verification Checklist

- [ ] Model directory exists: `models/deepseek-ocr/`
- [ ] Config file present: `config.json`
- [ ] Model code present: `modeling_deepseekocr.py`
- [ ] Tokenizer config present: `tokenizer_config.json`
- [ ] Preprocessor config present: `preprocessor_config.json`
- [ ] Safetensors weight files downloaded (several GB)
- [ ] Dependencies installed (including safetensors>=0.4.0)
- [ ] Test script runs without errors
- [ ] Pipeline processes sample document

## File Structure

After setup, your directory should look like:

```
deepseekocr/
├── models/
│   └── deepseek-ocr/
│       ├── config.json
│       ├── modeling_deepseekocr.py
│       ├── tokenizer_config.json
│       ├── preprocessor_config.json
│       ├── generation_config.json
│       ├── model.safetensors (or model-xxxxx-of-xxxxx.safetensors)
│       └── README.md
├── deepseek_ocr_pipeline.py
├── requirements.txt
├── README.md
└── MANUAL_SETUP.md
```

## Alternative: Using HuggingFace CLI

```bash
# Install huggingface-cli
pip install --user huggingface-hub

# Login (optional, for private models)
huggingface-cli login

# Download entire model repo
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir ./models/deepseek-ocr
```

## Notes

- Total download size: ~20-50GB depending on model variant
- First run may take extra time for model compilation
- GPU memory requirement: 16GB+ VRAM
- Model files are cached; no need to re-download on subsequent runs

