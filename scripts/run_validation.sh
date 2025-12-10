#!/bin/bash
# Validation launcher script for SQL Codegen SLM
# Module 2.3: Training Validation & Smoke Testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🔍 SQL Codegen SLM - Training Validation"
echo "========================================="
echo ""

# Check if running in Colab
if [[ -n "$COLAB_GPU" ]] || [[ -d "/content" ]]; then
    echo "✅ Running in Google Colab"
    IN_COLAB=true
    CONFIG_PATH="${1:-$PROJECT_DIR/training/configs/test_config.yaml}"
else
    echo "📍 Running locally"
    IN_COLAB=false
    CONFIG_PATH="${1:-$PROJECT_DIR/training/configs/test_config.yaml}"
fi

# Check GPU availability
echo ""
echo "🔍 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "   ✅ GPU: $GPU_NAME ($GPU_MEM)"
else
    echo "   ⚠️ nvidia-smi not found"
    echo "   Validation requires GPU!"
    if [[ "$IN_COLAB" == "false" ]]; then
        echo "   Running CPU-only validation (limited)..."
    fi
fi

# Check Python environment
echo ""
echo "🔍 Checking Python environment..."
python --version
if python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
    echo "   ✅ PyTorch available"
else
    echo "   ❌ PyTorch not found"
    exit 1
fi

if python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null; then
    echo "   ✅ Transformers available"
else
    echo "   ❌ Transformers not found"
    exit 1
fi

# Check config file
echo ""
echo "🔍 Checking configuration..."
if [[ -f "$CONFIG_PATH" ]]; then
    echo "   ✅ Config: $CONFIG_PATH"
else
    echo "   ❌ Config not found: $CONFIG_PATH"
    exit 1
fi

# Check data files (if in Colab)
if [[ "$IN_COLAB" == "true" ]]; then
    echo ""
    echo "🔍 Checking data files..."
    if [[ -f "/content/data/train_postgres.jsonl" ]]; then
        TRAIN_COUNT=$(wc -l < "/content/data/train_postgres.jsonl")
        echo "   ✅ train_postgres.jsonl: $TRAIN_COUNT examples"
    else
        echo "   ⚠️ train_postgres.jsonl not found - will download from GCS"
    fi
fi

# Run validation
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Running validation suite..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "$PROJECT_DIR"

# Run validation
python -m training.validation "$CONFIG_PATH"
VALIDATION_EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ $VALIDATION_EXIT_CODE -eq 0 ]]; then
    echo "✅ VALIDATION PASSED"
    echo ""
    echo "Ready for full training!"
    echo "Run: python -m training.train --config training/configs/mistral_lora_config.yaml"
else
    echo "❌ VALIDATION FAILED"
    echo ""
    echo "Please fix the issues above before running full training."
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $VALIDATION_EXIT_CODE
