#!/usr/bin/env python3
"""
DeepSeek OCR Pipeline
End-to-end pipeline for processing documents with images, text, and tables.
"""

import os
import json
import argparse
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    from PIL import Image
    from pdf2image import convert_from_path
except ImportError as e:
    print(json.dumps({"error": f"Missing required package: {str(e)}"}))
    sys.exit(1)


class DeepSeekOCRPipeline:
    """Main pipeline class for DeepSeek OCR processing."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf', '.doc', '.docx'}
    
    def __init__(self, model_name: str = 'deepseek-ai/DeepSeek-OCR'):
        """Initialize the OCR pipeline with DeepSeek model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True,
                use_safetensors=True
            )
            self.model = self.model.eval().cuda().to(torch.bfloat16)
            self.temp_dir = tempfile.mkdtemp(prefix='deepseek_ocr_')
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def __del__(self):
        """Cleanup temporary directory on deletion."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate input file format and existence."""
        path = Path(file_path)
        
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return {"error": "Unsupported file type.", "supported": list(self.SUPPORTED_FORMATS)}
        
        return {"valid": True}
    
    def parse_markdown_to_blocks(self, markdown_text: str, coordinates: List = None) -> List[Dict[str, Any]]:
        """Parse markdown output into structured blocks."""
        blocks = []
        lines = markdown_text.split('\n')
        current_block = {"type": "text", "content": [], "confidence": 0.95}
        in_table = False
        table_rows = []
        
        for line in lines:
            # Detect table rows (markdown format: | col1 | col2 |)
            if line.strip().startswith('|') and line.strip().endswith('|'):
                if not in_table:
                    if current_block["content"]:
                        blocks.append({
                            "type": current_block["type"],
                            "content": '\n'.join(current_block["content"]).strip(),
                            "coordinates": coordinates or [0, 0, 0, 0],
                            "confidence": current_block["confidence"]
                        })
                        current_block = {"type": "text", "content": [], "confidence": 0.95}
                    in_table = True
                    table_rows = []
                
                # Parse table row
                cells = [cell.strip() for cell in line.strip('|').split('|')]
                # Skip separator rows (e.g., |---|---|)
                if not all(re.match(r'^-+$', cell.strip()) for cell in cells):
                    table_rows.append(cells)
            else:
                if in_table:
                    # End of table, save it
                    blocks.append({
                        "type": "table",
                        "content": self._format_table_as_text(table_rows),
                        "coordinates": coordinates or [0, 0, 0, 0],
                        "confidence": 0.90,
                        "table_data": table_rows
                    })
                    in_table = False
                    table_rows = []
                
                # Regular text line
                if line.strip():
                    # Detect potential image references
                    if line.strip().startswith('![') or '<image>' in line.lower():
                        if current_block["content"]:
                            blocks.append({
                                "type": current_block["type"],
                                "content": '\n'.join(current_block["content"]).strip(),
                                "coordinates": coordinates or [0, 0, 0, 0],
                                "confidence": current_block["confidence"]
                            })
                            current_block = {"type": "text", "content": [], "confidence": 0.95}
                        
                        blocks.append({
                            "type": "image",
                            "content": line.strip(),
                            "coordinates": coordinates or [0, 0, 0, 0],
                            "confidence": 0.85
                        })
                    else:
                        current_block["content"].append(line)
        
        # Handle any remaining table
        if in_table and table_rows:
            blocks.append({
                "type": "table",
                "content": self._format_table_as_text(table_rows),
                "coordinates": coordinates or [0, 0, 0, 0],
                "confidence": 0.90,
                "table_data": table_rows
            })
        
        # Handle any remaining text block
        if current_block["content"]:
            blocks.append({
                "type": current_block["type"],
                "content": '\n'.join(current_block["content"]).strip(),
                "coordinates": coordinates or [0, 0, 0, 0],
                "confidence": current_block["confidence"]
            })
        
        return blocks
    
    def _format_table_as_text(self, table_data: List[List[str]]) -> str:
        """Format table data as readable text."""
        if not table_data:
            return ""
        return '\n'.join([' | '.join(row) for row in table_data])
    
    def _check_libreoffice(self) -> bool:
        """Check if LibreOffice is installed."""
        try:
            result = subprocess.run(
                ['soffice', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _convert_doc_to_pdf(self, doc_path: str) -> Optional[str]:
        """Convert .doc or .docx file to PDF using LibreOffice."""
        if not self._check_libreoffice():
            raise RuntimeError(
                "LibreOffice not found. Install LibreOffice to process .doc/.docx files: "
                "brew install libreoffice (macOS) or apt-get install libreoffice (Linux)"
            )
        
        try:
            output_dir = self.temp_dir
            # Convert to PDF using LibreOffice headless mode
            subprocess.run(
                [
                    'soffice',
                    '--headless',
                    '--convert-to', 'pdf',
                    '--outdir', output_dir,
                    doc_path
                ],
                capture_output=True,
                timeout=60,
                check=True
            )
            
            # Find the generated PDF
            doc_filename = Path(doc_path).stem
            pdf_path = Path(output_dir) / f"{doc_filename}.pdf"
            
            if pdf_path.exists():
                return str(pdf_path)
            else:
                raise RuntimeError(f"PDF conversion failed for {doc_path}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Document conversion timed out: {doc_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Document conversion failed: {e.stderr.decode()}")
    
    def _convert_pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to images."""
        try:
            images = convert_from_path(pdf_path, dpi=300)
            image_paths = []
            
            for i, image in enumerate(images):
                image_path = Path(self.temp_dir) / f"page_{i+1}.png"
                image.save(str(image_path), 'PNG')
                image_paths.append(str(image_path))
            
            return image_paths
        except Exception as e:
            raise RuntimeError(f"PDF to image conversion failed: {str(e)}")
    
    def _prepare_file_for_ocr(self, file_path: str) -> List[str]:
        """Prepare file for OCR processing, converting Word docs if needed.
        
        Returns list of image paths to process.
        """
        file_ext = Path(file_path).suffix.lower()
        
        # Word documents need conversion
        if file_ext in {'.doc', '.docx'}:
            # Convert to PDF first
            pdf_path = self._convert_doc_to_pdf(file_path)
            # Then convert PDF to images
            return self._convert_pdf_to_images(pdf_path)
        
        # PDF files can be converted to images for better processing
        elif file_ext == '.pdf':
            return self._convert_pdf_to_images(file_path)
        
        # Image files can be processed directly
        else:
            return [file_path]
    
    def process_file(self, file_path: str, output_dir: str = None) -> Dict[str, Any]:
        """Process a single file through OCR pipeline."""
        # Validate file
        validation = self.validate_file(file_path)
        if "error" in validation:
            return validation
        
        try:
            # Set up output directory
            if output_dir is None:
                output_dir = Path(file_path).parent / "ocr_output"
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare file for OCR (convert Word docs to images if needed)
            image_paths = self._prepare_file_for_ocr(file_path)
            
            # Prepare prompt for OCR
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            
            # Process each page/image
            all_pages = []
            for page_num, image_path in enumerate(image_paths, start=1):
                # Run OCR inference
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=image_path,
                    output_path=str(output_dir),
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    save_results=True,
                    test_compress=True
                )
                
                # Parse result into structured format
                markdown_content = result.get('text', '')
                blocks = self.parse_markdown_to_blocks(markdown_content)
                
                all_pages.append({
                    "page_number": page_num,
                    "blocks": blocks
                })
            
            # Structure output according to schema
            output = {
                "pages": all_pages
            }
            
            return output
            
        except torch.cuda.OutOfMemoryError:
            return {"error": "GPU out of memory. Try reducing image size or batch size."}
        except RuntimeError as e:
            return {"error": str(e), "file": file_path}
        except Exception as e:
            return {"error": f"OCR failed: {str(e)}", "file": file_path}
    
    def process_batch(self, file_paths: List[str], output_dir: str = None) -> Dict[str, Any]:
        """Process multiple files in batch."""
        results = {}
        
        for file_path in file_paths:
            file_name = Path(file_path).name
            result = self.process_file(file_path, output_dir)
            results[file_name] = result
        
        return results


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description='DeepSeek OCR Pipeline for document processing'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input file or directory containing files to process'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for results (default: ./ocr_output)'
    )
    parser.add_argument(
        '--json-output',
        type=str,
        default=None,
        help='Path to save JSON output (default: stdout)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    try:
        pipeline = DeepSeekOCRPipeline()
    except Exception as e:
        print(json.dumps({"error": f"Failed to initialize pipeline: {str(e)}"}))
        sys.exit(1)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = pipeline.process_file(str(input_path), args.output)
    elif input_path.is_dir():
        files = [
            str(f) for f in input_path.iterdir()
            if f.suffix.lower() in pipeline.SUPPORTED_FORMATS
        ]
        if not files:
            result = {"error": "No supported files found in directory"}
        else:
            result = pipeline.process_batch(files, args.output)
    else:
        result = {"error": f"Input path not found: {args.input}"}
    
    # Output result
    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    
    if args.json_output:
        with open(args.json_output, 'w', encoding='utf-8') as f:
            f.write(output_json)
        print(f"Results saved to {args.json_output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()

