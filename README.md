# DeepSeek OCR Pipeline

An end-to-end document processing pipeline using DeepSeek OCR that extracts and structures content from files containing images, text, and tables into JSON format for downstream consumption.

## What This Does

This pipeline processes document files through DeepSeek OCR to extract structured content. The program analyzes input documents, identifies content blocks (text, tables, images), and outputs parsed data in standardized JSON format suitable for vector storage and data chunking applications.

## Usage

```bash
# Process single file
python deepseek_ocr_pipeline.py input_document.jpg -o output_directory

# Process directory of files
python deepseek_ocr_pipeline.py documents/ -o output_directory --json-output results.json

# Output to stdout
python deepseek_ocr_pipeline.py input.pdf
```

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
          ]
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

**Data Ordering:**
Blocks appear in the original document reading order (top-to-bottom, left-to-right).

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

## System Requirements

**Minimum Hardware:**
- NVIDIA GPU with 16GB+ VRAM (A100-40G recommended for production)
- 32GB+ system RAM
- CUDA 11.8 compatible GPU

**Software:**
- Python 3.12.9
- CUDA 11.8
- Linux or macOS (Windows not officially supported)

## Installation

```bash
# Install CUDA 11.8 first (system-specific)

# Install Python dependencies
pip install torch==2.6.0 transformers==4.46.3 tokenizers==0.20.3 einops addict easydict flash-attn==2.7.3 --no-build-isolation Pillow

# Or use requirements file
pip install -r requirements.txt --no-build-isolation
```

## Supported File Types

- Images: .jpg, .jpeg, .png, .bmp, .tiff
- Documents: .pdf

## Reference

- Model: [DeepSeek-OCR on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- Paper: [arXiv:2510.18234](https://arxiv.org/pdf/2510.18234)

