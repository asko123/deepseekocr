#!/usr/bin/env python3
"""
DeepSeek OCR Pipeline
End-to-end pipeline for processing documents with images, text, and tables.
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    from PIL import Image
except ImportError as e:
    print(json.dumps({"error": f"Missing required package: {str(e)}"}))
    sys.exit(1)


class DeepSeekOCRPipeline:
    """Main pipeline class for DeepSeek OCR processing."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf'}
    
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
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
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
            
            # Prepare prompt for OCR
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            
            # Run OCR inference
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=str(file_path),
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
            
            # Structure output according to schema
            output = {
                "pages": [
                    {
                        "page_number": 1,
                        "blocks": blocks
                    }
                ]
            }
            
            return output
            
        except torch.cuda.OutOfMemoryError:
            return {"error": "GPU out of memory. Try reducing image size or batch size."}
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

