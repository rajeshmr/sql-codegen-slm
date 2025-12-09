#!/bin/bash
# Convert SQLite syntax to PostgreSQL in Mistral training data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🔄 SQLite to PostgreSQL Converter"
echo "=================================="
echo ""

# Check if Mistral JSONL files exist
TRAIN_MISTRAL="$PROJECT_DIR/data/processed/train_mistral.jsonl"
if [[ ! -f "$TRAIN_MISTRAL" ]]; then
    echo "❌ Error: Mistral training data not found"
    echo "   Run ./scripts/convert_to_mistral.sh first"
    exit 1
fi

# Try to use conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^sql-codegen "; then
        conda activate sql-codegen 2>/dev/null || true
    fi
fi

# Run the Python converter
cd "$PROJECT_DIR"
python -m data.postgres_converter

echo ""
echo "✅ PostgreSQL conversion complete!"
echo ""
echo "Next steps:"
echo "  1. Inspect output: head -1 data/processed/train_postgres.jsonl | python -m json.tool"
echo "  2. Compare before/after: diff <(sed -n '1p' data/processed/train_mistral.jsonl | python -m json.tool) <(sed -n '1p' data/processed/train_postgres.jsonl | python -m json.tool)"
echo "  3. Run tests: pytest tests/data/test_postgres_converter.py -v"
echo "  4. Proceed to Module 1.5: Create train/val/test splits"
