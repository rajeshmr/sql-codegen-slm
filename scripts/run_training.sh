#!/bin/bash
# Training launcher script for SQL Codegen SLM
# Module 2.2: Training Pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 SQL Codegen SLM Training Launcher"
echo "====================================="
echo ""

# Check if running in Colab
if [[ -n "$COLAB_GPU" ]] || [[ -d "/content" ]]; then
    echo "✅ Running in Google Colab"
    IN_COLAB=true
else
    echo "📍 Running locally"
    IN_COLAB=false
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
    echo "   Training will be very slow without GPU!"
fi

# Check GCS authentication (if in Colab)
if [[ "$IN_COLAB" == "true" ]]; then
    echo ""
    echo "🔍 Checking GCS authentication..."
    if gsutil ls gs://sql-codegen-slm-data &> /dev/null; then
        echo "   ✅ GCS bucket accessible"
    else
        echo "   ❌ Cannot access GCS bucket"
        echo "   Run: from google.colab import auth; auth.authenticate_user()"
        exit 1
    fi
fi

# Check data files
echo ""
echo "🔍 Checking data files..."
if [[ "$IN_COLAB" == "true" ]]; then
    DATA_DIR="/content/data"
else
    DATA_DIR="$PROJECT_DIR/data/processed"
fi

if [[ -f "$DATA_DIR/train_postgres.jsonl" ]]; then
    TRAIN_COUNT=$(wc -l < "$DATA_DIR/train_postgres.jsonl")
    echo "   ✅ train_postgres.jsonl: $TRAIN_COUNT examples"
else
    echo "   ❌ train_postgres.jsonl not found at $DATA_DIR"
    exit 1
fi

if [[ -f "$DATA_DIR/val_postgres.jsonl" ]]; then
    VAL_COUNT=$(wc -l < "$DATA_DIR/val_postgres.jsonl")
    echo "   ✅ val_postgres.jsonl: $VAL_COUNT examples"
else
    echo "   ❌ val_postgres.jsonl not found at $DATA_DIR"
    exit 1
fi

# Configuration
CONFIG_FILE="${1:-$PROJECT_DIR/training/configs/mistral_lora_config.yaml}"
RESUME_FLAG="${2:-}"

echo ""
echo "📋 Configuration"
echo "   Config: $CONFIG_FILE"
echo "   Resume: ${RESUME_FLAG:-No}"

# Print timing info
echo ""
echo "⏱️ Timing"
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "   Start time: $START_TIME"
echo "   Estimated duration: 8-12 hours (A100)"
echo ""

# Launch training
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting training..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "$PROJECT_DIR"

if [[ -n "$RESUME_FLAG" ]]; then
    python -m training.train --config "$CONFIG_FILE" --resume
else
    python -m training.train --config "$CONFIG_FILE"
fi

# Print completion info
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "✅ Training completed!"
echo "   Start: $START_TIME"
echo "   End:   $END_TIME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
