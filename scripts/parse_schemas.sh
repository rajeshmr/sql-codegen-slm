#!/bin/bash
# Parse Spider schema files and create index

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🕷️  Spider Schema Parser"
echo "========================"
echo ""

# Create output directory
SCHEMAS_DIR="$PROJECT_DIR/data/processed/schemas"
if [[ ! -d "$SCHEMAS_DIR" ]]; then
    echo "📁 Creating directory: $SCHEMAS_DIR"
    mkdir -p "$SCHEMAS_DIR"
fi

# Check if Spider dataset exists
SPIDER_DB_DIR="$PROJECT_DIR/data/raw/spider/database"
if [[ ! -d "$SPIDER_DB_DIR" ]]; then
    echo "❌ Error: Spider database directory not found: $SPIDER_DB_DIR"
    echo "   Run ./scripts/download_spider.sh first"
    exit 1
fi

# Run the Python schema parser
echo "Running schema parser..."
cd "$PROJECT_DIR"

# Try to use conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^sql-codegen "; then
        conda activate sql-codegen 2>/dev/null || true
    fi
fi

python -m data.schema_parser

echo ""
echo "✅ Schema parsing complete!"
echo ""
echo "Next steps:"
echo "  1. Check schemas: ls $SCHEMAS_DIR | head"
echo "  2. View index: cat $SCHEMAS_DIR/schema_index.json | head -50"
echo "  3. Run tests: pytest tests/data/test_schema_parser.py -v"
