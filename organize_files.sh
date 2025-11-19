#!/bin/bash
# Script to organize model files into deepseek-ocr directory

echo "Organizing model files..."

# Create deepseek-ocr directory if it doesn't exist
mkdir -p deepseek-ocr

# Move model files (but not pipeline scripts)
echo "Moving model files to deepseek-ocr/..."

# Move safetensors weight files
mv *.safetensors deepseek-ocr/ 2>/dev/null && echo "✓ Moved weight files"

# Move JSON config files (but keep README.md and other docs)
mv config.json tokenizer_config.json preprocessor_config.json generation_config.json deepseek-ocr/ 2>/dev/null && echo "✓ Moved config files"

# Move model Python files
mv modeling_*.py configuration_*.py deepencoder.py conversation.py deepseek-ocr/ 2>/dev/null && echo "✓ Moved model code files"

echo ""
echo "Verification:"
ls -la deepseek-ocr/ | head -20

echo ""
echo "Model files organized! Now you can run:"
echo "  python deepseek_ocr_pipeline.py document.pdf --no-flash-attn"

