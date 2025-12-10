#!/bin/bash
# Finalize data pipeline: create splits, generate demo schemas, compute statistics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🎯 Finalizing Data Pipeline"
echo "============================"
echo ""

# Check if PostgreSQL JSONL files exist
TRAIN_POSTGRES="$PROJECT_DIR/data/processed/train_postgres.jsonl"
DEV_POSTGRES="$PROJECT_DIR/data/processed/dev_postgres.jsonl"

if [[ ! -f "$TRAIN_POSTGRES" ]]; then
    echo "❌ Error: Training data not found"
    echo "   Run ./scripts/convert_to_postgres.sh first"
    exit 1
fi

if [[ ! -f "$DEV_POSTGRES" ]]; then
    echo "❌ Error: Dev data not found"
    echo "   Run ./scripts/convert_to_postgres.sh first"
    exit 1
fi

# Try to use conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^sql-codegen "; then
        conda activate sql-codegen 2>/dev/null || true
    fi
fi

cd "$PROJECT_DIR"

# Step 1: Create train/validation/test splits
echo "📊 Step 1: Creating data splits..."
python -m data.split_creator

# Step 2: Generate demo schemas
echo ""
echo "📝 Step 2: Generating demo schemas..."
python -m data.demo_schema_generator

# Step 3: Compute comprehensive statistics
echo ""
echo "📈 Step 3: Computing dataset statistics..."
python -m data.dataset_stats

# Final summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ DATA PIPELINE COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Output files:"
echo "  📁 data/processed/"
echo "     ├── train_postgres.jsonl  (training set)"
echo "     ├── val_postgres.jsonl    (validation set)"
echo "     ├── test_postgres.jsonl   (test set)"
echo "     ├── split_info.json       (split metadata)"
echo "     └── dataset_statistics.json"
echo ""
echo "  📁 data/demo/"
echo "     ├── ecommerce_schema.sql"
echo "     ├── finance_schema.sql"
echo "     ├── healthcare_schema.sql"
echo "     ├── saas_schema.sql"
echo "     ├── retail_schema.sql"
echo "     └── demo_schemas.json"
echo ""
echo "Next steps:"
echo "  1. Run tests: pytest tests/data/test_final_pipeline.py -v"
echo "  2. Review statistics: cat data/processed/dataset_statistics.json | python -m json.tool"
echo "  3. Proceed to Module 2: Model Training 🚀"
