# Complex Table Support

## Overview

The DeepSeek OCR Pipeline includes enhanced support for complex table structures commonly found in business documents, research papers, and reports.

## Supported Complex Structures

### 1. Merged Cells (Colspan and Rowspan)
- **Horizontal merges**: Multiple columns merged into one cell
- **Vertical merges**: Multiple rows merged into one cell
- Detected from .docx files using gridSpan and vMerge properties
- Preserved in table_data output

### 2. Multi-Level Headers
- Headers spanning multiple rows
- Hierarchical column groupings
- Nested header structures

### 3. Irregular Table Structures
- Tables with varying numbers of columns per row
- Mixed content cells (text, numbers, formulas)
- Empty cells indicating structural elements

### 4. Large Tables
- Multi-page tables automatically split by page
- Tables with hundreds of rows and columns
- Cells with extensive content (>200 characters)

## Detection and Metadata

The pipeline automatically analyzes each table and provides metadata:

```json
"table_metadata": {
  "rows": 15,
  "columns": 8,
  "is_complex": true,
  "has_merged_cells": true
}
```

### Complexity Indicators

A table is marked as complex if:
- Row lengths vary (indication of merged cells)
- More than 10% of cells are empty (structural indicators)
- Cells contain multiline content or >200 characters
- Detected merged cell properties in source document

## File Format Handling

### .docx Files (Best Support)
- Direct extraction of cell merge information
- Preserves colspan and rowspan data
- Maintains table structure metadata
- Handles nested tables within cells

### PDF Files
- Tables converted to images at 300 DPI
- OCR recognizes complex layouts
- Markdown representation may simplify structure
- Some merge information inferred from spacing

### Image Files
- Relies entirely on DeepSeek OCR capabilities
- Complex structures detected visually
- May simplify very complex layouts
- Best results with high-resolution images

## Output Format

Complex tables are output in two representations:

### 1. Readable Text Format (`content` field)
```
Header1 | Header2 | Header3
Data1   | Data2   | Data3
```

### 2. Structured Data (`table_data` field)
```json
[
  ["Header1", "Header2", "Header3"],
  ["Data1", "Data2", "Data3"]
]
```

## Limitations

While the pipeline handles complex tables well, some limitations exist:

1. **Nested Tables**: Tables within table cells are flattened
2. **Complex Formatting**: Cell styling, colors, borders not preserved
3. **Formulas**: Only formula results extracted, not formulas themselves
4. **Very Complex Layouts**: Extremely irregular structures may be simplified

## Best Practices

For optimal complex table processing:

1. **Use .docx when possible** - provides best structural information
2. **High-resolution images** - 300+ DPI for scanned documents
3. **Clean layouts** - avoid excessive formatting complexity
4. **Validate output** - check `is_complex` flag for manual review needs
5. **Monitor confidence** - low confidence scores indicate potential issues

## Example: Complex Table Processing

Input: Financial report with merged header cells and multi-row data

```
| Quarter | Revenue      | Expenses     |
|         | Q1    | Q2  | Q1    | Q2   |
| 2024    | $100K | $120K | $80K | $90K |
```

Output:
```json
{
  "type": "table",
  "table_data": [
    ["Quarter", "Revenue", "Revenue", "Expenses", "Expenses"],
    ["", "Q1", "Q2", "Q1", "Q2"],
    ["2024", "$100K", "$120K", "$80K", "$90K"]
  ],
  "table_metadata": {
    "rows": 3,
    "columns": 5,
    "is_complex": true,
    "has_merged_cells": true
  }
}
```

## Troubleshooting

**Table not detected**: Ensure clear borders or spacing between cells
**Incorrect structure**: Try higher resolution or .docx format
**Missing merged cells**: Check if original document properly defines merges
**Low confidence**: Complex tables may need manual validation

