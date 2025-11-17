# Manual Download Guide for DeepSeek OCR Model Files

This guide shows you how to manually download model files through your web browser and copy them to the correct location.

## Step 1: Create the Model Directory

Open your terminal and run:

```bash
cd /Users/Tawfiq/Desktop/deepseekocr
mkdir -p models/deepseek-ocr
```

This creates: `/Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr/`

## Step 2: Download Files via Web Browser

### A. Configuration Files (Small, text files)

Visit: https://huggingface.co/deepseek-ai/DeepSeek-OCR/tree/main

For each file below:
1. Click the filename in the browser
2. Click the "Download" button (⬇️ icon) in the top-right
3. Save to your Downloads folder

**Files to download:**
- `config.json`
- `modeling_deepseekocr.py`
- `tokenizer_config.json`
- `preprocessor_config.json`
- `generation_config.json`

### B. Model Weight Files (.safetensors)

**Important:** These are LARGE files (several GB each). Download may take time.

1. Go to: https://huggingface.co/deepseek-ai/DeepSeek-OCR/tree/main
2. Look for files ending in `.safetensors`
3. You'll see either:
   - One file: `model.safetensors` (single large file)
   - Multiple files: `model-00001-of-00004.safetensors`, etc. (sharded)

**For each .safetensors file:**
1. Click the filename
2. Click the "Download" button (⬇️)
3. Wait for download to complete (may take 10-30 minutes per file)

**If you see multiple shards, download ALL of them:**
- `model-00001-of-00004.safetensors`
- `model-00002-of-00004.safetensors`
- `model-00003-of-00004.safetensors`
- `model-00004-of-00004.safetensors`

(The number of shards may vary - download all that are present)

## Step 3: Move Files to Model Directory

### Option A: Using Finder (macOS GUI)

1. Open Finder
2. Press `Cmd + Shift + G` (Go to folder)
3. Type: `/Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr`
4. Press Enter
5. Open another Finder window to your Downloads folder
6. Select all downloaded files
7. Drag them into the model directory
8. Wait for large files to copy

### Option B: Using Terminal (Command Line)

```bash
# Navigate to Downloads
cd ~/Downloads

# Move config files to model directory
mv config.json /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr/
mv modeling_deepseekocr.py /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr/
mv tokenizer_config.json /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr/
mv preprocessor_config.json /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr/
mv generation_config.json /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr/

# Move safetensors files (adjust names based on what you downloaded)
# For single file:
mv model.safetensors /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr/

# OR for sharded files:
mv model-*.safetensors /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr/

# Or move each shard individually:
# mv model-00001-of-00004.safetensors /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr/
# mv model-00002-of-00004.safetensors /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr/
# mv model-00003-of-00004.safetensors /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr/
# mv model-00004-of-00004.safetensors /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr/
```

## Step 4: Verify Files Are in Place

```bash
cd /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr
ls -lh

# You should see:
# config.json
# modeling_deepseekocr.py
# tokenizer_config.json
# preprocessor_config.json
# generation_config.json
# model.safetensors (or multiple model-*.safetensors files)
```

Check safetensors file sizes:

```bash
ls -lh *.safetensors
# Each file should be several GB (e.g., 4.8G, 9.2G)
```

## Step 5: Run Verification

```bash
cd /Users/Tawfiq/Desktop/deepseekocr
python test_model_load.py
```

Expected output:
```
✓ Model directory found: ./models/deepseek-ocr
✓ Found: config.json (Model configuration)
✓ Found: modeling_deepseekocr.py (Custom model code)
✓ Found: tokenizer_config.json (Tokenizer configuration)
✓ Found: preprocessor_config.json (Preprocessor configuration)
✓ Found X safetensors weight file(s)
  - model-00001-of-00004.safetensors (4.85 GB)
  - model-00002-of-00004.safetensors (4.89 GB)
  - model-00003-of-00004.safetensors (4.91 GB)
  - model-00004-of-00004.safetensors (3.52 GB)
  Total size: 18.17 GB
✓ Tokenizer loaded
✓ Model loaded
✓ Model moved to GPU
✓ ALL TESTS PASSED!
```

## Troubleshooting

### Issue: "File too large to download"

Some browsers have issues with very large files. Try:
1. Use a download manager (e.g., Free Download Manager, JDownloader)
2. Use `wget` or `curl` in terminal:
   ```bash
   cd /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr
   wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00001-of-00004.safetensors
   ```
3. Use the HuggingFace CLI (recommended for large files):
   ```bash
   pip install --user huggingface-hub
   huggingface-cli download deepseek-ai/DeepSeek-OCR --include "*.safetensors" --local-dir ./models/deepseek-ocr
   ```

### Issue: "Download interrupted"

Resume the download:
```bash
cd /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr
wget -c https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00001-of-00004.safetensors
# The -c flag resumes interrupted downloads
```

### Issue: "Not enough disk space"

Model files are ~20-25GB total. Check available space:
```bash
df -h ~/Desktop
```

Free up space if needed, or download to an external drive.

### Issue: "File won't copy - permission denied"

Make sure the destination directory is writable:
```bash
chmod -R u+w /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr/
```

### Issue: "Browser shows 'Download quota exceeded'"

HuggingFace has download limits. Wait a few hours or try:
1. Create a free HuggingFace account and login
2. Use Git LFS (better for large files):
   ```bash
   cd /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr
   git lfs install
   git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR .
   ```

## Quick Reference: File Checklist

After copying, your directory should contain:

```
models/deepseek-ocr/
├── config.json                           ✓ (text file, few KB)
├── modeling_deepseekocr.py              ✓ (text file, ~100 KB)
├── tokenizer_config.json                ✓ (text file, few KB)
├── preprocessor_config.json             ✓ (text file, few KB)
├── generation_config.json               ✓ (text file, few KB)
└── model-*.safetensors                  ✓ (binary files, several GB each)
```

**Total size:** Approximately 20-25 GB

## Using wget for All Files (Alternative)

If you prefer command line download:

```bash
cd /Users/Tawfiq/Desktop/deepseekocr/models/deepseek-ocr

# Download config files
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/config.json
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/modeling_deepseekocr.py
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/tokenizer_config.json
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/preprocessor_config.json
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/generation_config.json

# Check the HuggingFace page for exact safetensors filenames, then:
# wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00001-of-00004.safetensors
# wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00002-of-00004.safetensors
# ... (continue for all shards)
```

## Tips for Faster Downloads

1. **Use a wired connection** instead of WiFi
2. **Download during off-peak hours** (late night/early morning)
3. **Use multiple terminal windows** to download shards in parallel
4. **Consider Git LFS** - it handles large files better than web browser
5. **Use HuggingFace CLI** - more reliable for large files than browser

## Next Steps

Once all files are in place and verification passes:

```bash
# Test with a sample document
python deepseek_ocr_pipeline.py sample_document.jpg
```

You're ready to go!

