#!/bin/bash
# Convert Spider examples to Mistral instruction format

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🔄 Spider to Mistral Format Converter"
echo "======================================"
echo ""

# Check if Spider dataset exists
SPIDER_DIR="$PROJECT_DIR/data/raw/spider"
if [[ ! -f "$SPIDER_DIR/train_spider.json" ]]; then
    echo "❌ Error: Spider dataset not found"
    echo "   Run ./scripts/download_spider.sh first"
    exit 1
fi

# Check if schema index exists
SCHEMA_INDEX="$PROJECT_DIR/data/processed/schemas/schema_index.json"
if [[ ! -f "$SCHEMA_INDEX" ]]; then
    echo "❌ Error: Schema index not found"
    echo "   Run ./scripts/parse_schemas.sh first"
    exit 1
fi

# Create output directory
mkdir -p "$PROJECT_DIR/data/processed"

# Try to use conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^sql-codegen "; then
        conda activate sql-codegen 2>/dev/null || true
    fi
fi

# Run the Python converter
cd "$PROJECT_DIR"
python -m data.format_converter

echo ""
echo "✅ Conversion complete!"
echo ""
echo "Next steps:"
echo "  1. Inspect output: head -1 data/processed/train_mistral.jsonl | python -m json.tool"
echo "  2. Run tests: pytest tests/data/test_format_converter.py -v"
echo "  3. Proceed to Module 1.4: PostgreSQL conversion"
