# DeepSeek OCR Pipeline

An end-to-end document processing pipeline using DeepSeek OCR that extracts and structures content from files containing images, text, and tables into JSON format for downstream consumption.

## What This Does

This pipeline processes document files through DeepSeek OCR to extract structured content. The program analyzes input documents, identifies content blocks (text, tables, images), and outputs parsed data in standardized JSON format suitable for vector storage and data chunking applications.

## Usage

```bash
# Process single file (using local model)
python deepseek_ocr_pipeline.py input_document.jpg -o output_directory

# Specify custom model path
python deepseek_ocr_pipeline.py input.pdf --model-path ./models/deepseek-ocr

# Process directory of files
python deepseek_ocr_pipeline.py documents/ -o output_directory --json-output results.json

# Disable flash attention if unavailable
python deepseek_ocr_pipeline.py input.jpg --no-flash-attn

# Output to stdout
python deepseek_ocr_pipeline.py input.pdf
```

**CLI Options:**
- `input`: File or directory to process (required)
- `-o, --output`: Output directory (default: ./ocr_output)
- `--json-output`: Save results to JSON file
- `--model-path`: Path to model directory (default: ./models/deepseek-ocr)
- `--no-flash-attn`: Disable flash attention

## Output Format

The pipeline outputs JSON with the following schema:

```json
{
  "pages": [
    {
      "page_number": 1,
      "blocks": [
        {
          "type": "text",
          "content": "Extracted text content",
          "coordinates": [0, 0, 100, 50],
          "confidence": 0.95
        },
        {
          "type": "table",
          "content": "Header1 | Header2\nData1 | Data2",
          "coordinates": [0, 60, 100, 120],
          "confidence": 0.90,
          "table_data": [
            ["Header1", "Header2"],
            ["Data1", "Data2"]
          ],
          "table_metadata": {
            "rows": 2,
            "columns": 2,
            "is_complex": false,
            "has_merged_cells": false
          }
        },
        {
          "type": "image",
          "content": "![image description](path)",
          "coordinates": [0, 130, 100, 200],
          "confidence": 0.85
        }
      ]
    }
  ]
}
```

### Schema Details

**Field Specifications:**
- `pages`: Array of page objects (one per document page)
- `page_number`: Integer starting at 1
- `blocks`: Array of content blocks in reading order
- `type`: String enum: "text", "table", or "image"
- `content`: String containing extracted content
- `coordinates`: Array of 4 integers [left, top, right, bottom]
- `confidence`: Float between 0.0 and 1.0
- `table_data`: 2D array (list of lists) present only when type is "table"
- `table_metadata`: Object present only when type is "table" containing:
  - `rows`: Integer count of table rows
  - `columns`: Integer count of table columns
  - `is_complex`: Boolean indicating complex structure (merged cells, nested tables, etc.)
  - `has_merged_cells`: Boolean indicating presence of merged cells

**Data Ordering:**
Blocks appear in the original document reading order (top-to-bottom, left-to-right).

**Complex Table Support:**
The pipeline detects and handles complex table structures including merged cells, multi-row headers, and nested tables. Complex tables are flagged in `table_metadata.is_complex`.

## Error Reporting

Errors are returned in JSON format:

```json
{"error": "Unsupported file type.", "supported": [".jpg", ".png", ".pdf"]}
```

```json
{"error": "OCR failed: connection timeout", "file": "document.pdf"}
```

```json
{"error": "File not found: missing.jpg"}
```

```json
{"error": "GPU out of memory. Try reducing image size or batch size."}
```

```json
{"error": "LibreOffice not available for .doc conversion. Please convert .doc to .docx format", "file": "document.doc"}
```

## System Requirements

**Minimum Hardware:**
- NVIDIA GPU with 16GB+ VRAM (A100-40G recommended for production)
- 32GB+ system RAM
- CUDA 11.8 compatible GPU

**Software:**
- Python 3.12.9
- CUDA 11.8
- Linux or macOS (Windows not officially supported)
- No admin/root access required for core functionality

**Processing Methods:**
- .docx files: Uses python-docx library (no admin required)
- .doc files: Requires LibreOffice or conversion to .docx
- PDF files: Uses poppler-utils (can be installed locally)
- Images: Direct processing (no additional tools)

## Installation

### Step 1: Install System Dependencies

```bash
# Install CUDA 11.8 first (system-specific)

# Install Poppler for PDF processing (no admin required on Linux)
# For Linux without admin rights:
# Download poppler binaries to your home directory from https://poppler.freedesktop.org/

# For macOS:
brew install poppler

# For Ubuntu/Debian with admin:
sudo apt-get install poppler-utils
```

### Step 2: Install Python Dependencies

```bash
# Install Python dependencies (no admin required)
pip install --user torch==2.6.0 transformers==4.46.3 tokenizers==0.20.3 einops addict easydict flash-attn==2.7.3 Pillow pdf2image python-docx olefile --no-build-isolation

# Or use requirements file
pip install --user -r requirements.txt --no-build-isolation
```

### Step 3: Download Model Files

**Important:** Due to model size and access requirements, you must manually download the model files.

See [MANUAL_SETUP.md](MANUAL_SETUP.md) for complete step-by-step instructions.

**Quick Setup:**

```bash
# Create model directory
mkdir -p models/deepseek-ocr

# Option A: Clone with Git LFS (recommended)
cd models/deepseek-ocr
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR .

# Option B: Use HuggingFace CLI
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir ./models/deepseek-ocr

# Verify installation
python test_model_load.py
```

**Note:** LibreOffice is optional. The pipeline uses `python-docx` for .docx files (no admin required). LibreOffice provides better formatting if available but is not required.

## Supported File Types

- Images: .jpg, .jpeg, .png, .bmp, .tiff
- Documents: .pdf, .doc, .docx

## Documentation

- [MANUAL_SETUP.md](MANUAL_SETUP.md) - Complete guide for downloading and setting up model files
- [COMPLEX_TABLES.md](COMPLEX_TABLES.md) - Guide for handling complex table structures
- [test_model_load.py](test_model_load.py) - Verification script for model setup

## Reference

- Model: [DeepSeek-OCR on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- Model Code: [modeling_deepseekocr.py](https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/main/modeling_deepseekocr.py)
- Paper: [arXiv:2510.18234](https://arxiv.org/pdf/2510.18234)

