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
    from docx import Document as DocxDocument
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import _Cell, Table
    from docx.text.paragraph import Paragraph
except ImportError as e:
    print(json.dumps({"error": f"Missing required package: {str(e)}"}))
    sys.exit(1)

# Optional: olefile for .doc file support
try:
    import olefile
    OLEFILE_AVAILABLE = True
except ImportError:
    OLEFILE_AVAILABLE = False


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
        """Parse markdown output into structured blocks with enhanced table support."""
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
                
                # Parse table row with enhanced cell detection
                cells = [cell.strip() for cell in line.strip('|').split('|')]
                
                # Skip separator rows (e.g., |---|---|)
                if not all(re.match(r'^-+$', cell.strip()) for cell in cells):
                    # Detect potential merged cells (cells with extra spacing or special markers)
                    parsed_cells = []
                    for cell in cells:
                        # Check for colspan indicators (e.g., "text ^^" or empty cells)
                        if cell:
                            parsed_cells.append(cell)
                        else:
                            # Empty cell might indicate continuation of previous cell
                            parsed_cells.append("")
                    table_rows.append(parsed_cells)
            else:
                if in_table:
                    # End of table, save it with complexity analysis
                    complexity_info = self._analyze_table_complexity(table_rows)
                    table_block = {
                        "type": "table",
                        "content": self._format_table_as_text(table_rows),
                        "coordinates": coordinates or [0, 0, 0, 0],
                        "confidence": 0.90,
                        "table_data": table_rows,
                        "table_metadata": {
                            "rows": complexity_info["num_rows"],
                            "columns": complexity_info["num_cols"],
                            "is_complex": complexity_info["is_complex"],
                            "has_merged_cells": complexity_info["has_irregular_rows"] or complexity_info["empty_cell_ratio"] > 0.1
                        }
                    }
                    blocks.append(table_block)
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
            complexity_info = self._analyze_table_complexity(table_rows)
            table_block = {
                "type": "table",
                "content": self._format_table_as_text(table_rows),
                "coordinates": coordinates or [0, 0, 0, 0],
                "confidence": 0.90,
                "table_data": table_rows,
                "table_metadata": {
                    "rows": complexity_info["num_rows"],
                    "columns": complexity_info["num_cols"],
                    "is_complex": complexity_info["is_complex"],
                    "has_merged_cells": complexity_info["has_irregular_rows"] or complexity_info["empty_cell_ratio"] > 0.1
                }
            }
            blocks.append(table_block)
        
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
    
    def _analyze_table_complexity(self, table_data: List[List[str]]) -> Dict[str, Any]:
        """Analyze table structure to detect complexity."""
        if not table_data:
            return {"is_complex": False}
        
        num_rows = len(table_data)
        num_cols = len(table_data[0]) if table_data else 0
        
        # Check for irregular row lengths (indication of merged cells)
        row_lengths = [len(row) for row in table_data]
        has_irregular_rows = len(set(row_lengths)) > 1
        
        # Check for empty cells (might indicate merged cells)
        empty_cell_count = sum(1 for row in table_data for cell in row if not cell.strip())
        empty_cell_ratio = empty_cell_count / (num_rows * num_cols) if (num_rows * num_cols) > 0 else 0
        
        # Detect potential nested structure (cells with multiple lines or formatting)
        has_multiline_cells = any('\\n' in cell or len(cell) > 200 for row in table_data for cell in row)
        
        is_complex = has_irregular_rows or empty_cell_ratio > 0.1 or has_multiline_cells
        
        return {
            "is_complex": is_complex,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "has_irregular_rows": has_irregular_rows,
            "empty_cell_ratio": round(empty_cell_ratio, 2),
            "has_multiline_cells": has_multiline_cells
        }
    
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
    
    def _extract_docx_content(self, docx_path: str) -> str:
        """Extract content from .docx file as markdown-like text."""
        try:
            doc = DocxDocument(docx_path)
            content_parts = []
            
            for element in doc.element.body:
                if isinstance(element, CT_P):
                    # Paragraph
                    para = Paragraph(element, doc)
                    text = para.text.strip()
                    if text:
                        # Detect headings
                        if para.style.name.startswith('Heading'):
                            level = para.style.name.replace('Heading', '').strip()
                            if level.isdigit():
                                content_parts.append(f"{'#' * int(level)} {text}")
                            else:
                                content_parts.append(f"## {text}")
                        else:
                            content_parts.append(text)
                
                elif isinstance(element, CT_Tbl):
                    # Table - enhanced to handle complex structures
                    table = Table(element, doc)
                    table_text = []
                    table_structure = []
                    
                    for row_idx, row in enumerate(table.rows):
                        row_cells = []
                        row_data = []
                        for cell_idx, cell in enumerate(row.cells):
                            cell_text = cell.text.strip()
                            row_data.append(cell_text)
                            
                            # Check for merged cells by examining grid span
                            try:
                                tc = cell._element
                                tcPr = tc.get_or_add_tcPr()
                                grid_span = tcPr.gridSpan
                                v_merge = tcPr.vMerge
                                
                                cell_info = {
                                    "text": cell_text,
                                    "colspan": grid_span.val if grid_span is not None else 1,
                                    "rowspan_start": v_merge is not None and v_merge.val != "continue" if v_merge is not None else False,
                                    "rowspan_continue": v_merge is not None and (v_merge.val == "continue" or v_merge.val is None) if v_merge is not None else False
                                }
                                row_cells.append(cell_info)
                            except:
                                # Fallback for cells without merge info
                                row_cells.append({
                                    "text": cell_text,
                                    "colspan": 1,
                                    "rowspan_start": False,
                                    "rowspan_continue": False
                                })
                        
                        table_structure.append(row_cells)
                        table_text.append('| ' + ' | '.join(row_data) + ' |')
                    
                    if table_text:
                        content_parts.append('\n'.join(table_text))
            
            return '\n\n'.join(content_parts)
        except Exception as e:
            raise RuntimeError(f"Failed to extract content from .docx: {str(e)}")
    
    def _extract_doc_content(self, doc_path: str) -> str:
        """Extract content from .doc file (legacy format)."""
        if not OLEFILE_AVAILABLE:
            raise RuntimeError(
                "Processing .doc files requires 'olefile' package. "
                "Install it with: pip install olefile"
            )
        
        try:
            # Basic text extraction from .doc using olefile
            ole = olefile.OleFileIO(doc_path)
            
            # Try to read WordDocument stream
            if ole.exists('WordDocument'):
                # This is a simplified extraction - full .doc parsing is complex
                raise RuntimeError(
                    ".doc format requires conversion. Please convert to .docx or use LibreOffice. "
                    "Alternative: Use 'antiword' tool or convert file to .docx format."
                )
            else:
                raise RuntimeError("Invalid .doc file format")
        except Exception as e:
            raise RuntimeError(f"Failed to process .doc file: {str(e)}")
    
    def _convert_doc_to_pdf(self, doc_path: str) -> Optional[str]:
        """Convert .doc or .docx file to PDF using LibreOffice (if available)."""
        if not self._check_libreoffice():
            # LibreOffice not available, return None to trigger fallback
            return None
        
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
                return None
                
        except subprocess.TimeoutExpired:
            return None
        except subprocess.CalledProcessError:
            return None
    
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
    
    def _create_text_image(self, text: str, output_path: str) -> str:
        """Create an image from text content for OCR processing."""
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a white image
        img_width = 2480  # A4 width at 300 DPI
        img_height = 3508  # A4 height at 300 DPI
        img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Use default font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw text with wrapping
        margin = 100
        y_position = margin
        line_height = 30
        max_width = img_width - (2 * margin)
        
        for line in text.split('\n'):
            if not line.strip():
                y_position += line_height
                continue
            
            # Simple word wrapping
            words = line.split()
            current_line = ""
            
            for word in words:
                test_line = current_line + word + " "
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] - bbox[0] <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        draw.text((margin, y_position), current_line.strip(), fill='black', font=font)
                        y_position += line_height
                    current_line = word + " "
            
            if current_line:
                draw.text((margin, y_position), current_line.strip(), fill='black', font=font)
                y_position += line_height
            
            if y_position > img_height - margin:
                break
        
        img.save(output_path)
        return output_path
    
    def _prepare_file_for_ocr(self, file_path: str) -> tuple:
        """Prepare file for OCR processing, converting Word docs if needed.
        
        Returns tuple: (list of image paths, use_direct_text_extraction)
        """
        file_ext = Path(file_path).suffix.lower()
        
        # Word documents need conversion
        if file_ext == '.docx':
            # Try LibreOffice conversion first
            pdf_path = self._convert_doc_to_pdf(file_path)
            
            if pdf_path:
                # LibreOffice succeeded, use PDF conversion
                return (self._convert_pdf_to_images(pdf_path), False)
            else:
                # LibreOffice not available, use direct extraction
                try:
                    content = self._extract_docx_content(file_path)
                    # Create image from extracted text
                    image_path = Path(self.temp_dir) / "docx_content.png"
                    self._create_text_image(content, str(image_path))
                    return ([str(image_path)], True)
                except Exception as e:
                    raise RuntimeError(f"Failed to process .docx file: {str(e)}")
        
        elif file_ext == '.doc':
            # Try LibreOffice conversion first
            pdf_path = self._convert_doc_to_pdf(file_path)
            
            if pdf_path:
                return (self._convert_pdf_to_images(pdf_path), False)
            else:
                # .doc format is complex, provide helpful error
                raise RuntimeError(
                    "LibreOffice not available for .doc conversion. "
                    "Please convert .doc to .docx format, or install LibreOffice locally in your home directory. "
                    "Alternative: pip install --user libreoffice (if available as Python package)"
                )
        
        # PDF files can be converted to images for better processing
        elif file_ext == '.pdf':
            return (self._convert_pdf_to_images(file_path), False)
        
        # Image files can be processed directly
        else:
            return ([file_path], False)
    
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
            image_paths, direct_extraction = self._prepare_file_for_ocr(file_path)
            
            # Process each page/image
            all_pages = []
            
            if direct_extraction:
                # Content was directly extracted from Word doc
                # Still pass through OCR for consistency, or use extracted content directly
                for page_num, image_path in enumerate(image_paths, start=1):
                    # For direct extraction, we could skip OCR and use the text directly
                    # but we'll still process for consistency with the pipeline
                    try:
                        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
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
                        markdown_content = result.get('text', '')
                        blocks = self.parse_markdown_to_blocks(markdown_content)
                    except:
                        # If OCR fails on text image, use direct blocks
                        blocks = [{
                            "type": "text",
                            "content": "Content extracted from Word document (OCR unavailable)",
                            "coordinates": [0, 0, 0, 0],
                            "confidence": 0.80
                        }]
                    
                    all_pages.append({
                        "page_number": page_num,
                        "blocks": blocks
                    })
            else:
                # Standard OCR processing
                for page_num, image_path in enumerate(image_paths, start=1):
                    # Prepare prompt for OCR
                    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
                    
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

