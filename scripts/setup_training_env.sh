#!/bin/bash
# Setup Training Environment for SQL Codegen SLM
# Module 2.1: Training Environment Setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🔧 Setting up Training Environment"
echo "===================================="
echo ""

# Try to use conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^sql-codegen "; then
        echo "📦 Activating conda environment: sql-codegen"
        conda activate sql-codegen 2>/dev/null || true
    fi
fi

cd "$PROJECT_DIR"

# Check Python version
echo "🐍 Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
echo "   Python version: $PYTHON_VERSION"

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p training/logs
mkdir -p training/models
mkdir -p training/tensorboard
mkdir -p scripts/gcp

echo "   ✅ training/logs/"
echo "   ✅ training/models/"
echo "   ✅ training/tensorboard/"
echo "   ✅ scripts/gcp/"

# Install training requirements
echo ""
echo "📦 Installing training dependencies..."
echo "   This may take a few minutes..."

if [[ -f "training/requirements.txt" ]]; then
    pip install -r training/requirements.txt --quiet 2>&1 | while read line; do
        if [[ "$line" == *"Successfully"* ]]; then
            echo "   $line"
        fi
    done
    echo "   ✅ Dependencies installed"
else
    echo "   ❌ training/requirements.txt not found"
    exit 1
fi

# Run environment validation
echo ""
echo "🔍 Validating environment..."
python -m training.environment_setup

# Check wandb configuration
echo ""
echo "📊 Checking Weights & Biases (wandb) configuration..."
if command -v wandb &> /dev/null; then
    if wandb status 2>/dev/null | grep -q "logged in"; then
        echo "   ✅ wandb is configured"
    else
        echo "   ⚠️  wandb not logged in"
        echo ""
        echo "   To setup wandb (optional but recommended):"
        echo "   1. Create account at https://wandb.ai"
        echo "   2. Run: wandb login"
        echo "   3. Enter your API key"
    fi
else
    echo "   ⚠️  wandb command not found"
fi

# Final summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Training Environment Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Review config: training/configs/mistral_lora_config.yaml"
echo "  2. For GCP training:"
echo "     ./scripts/gcp/create_training_instance.sh"
echo "     ./scripts/gcp/sync_data_to_gcp.sh"
echo "     ./scripts/gcp/connect_to_instance.sh"
echo "  3. For local testing (requires GPU):"
echo "     python -m training.train --test-run"
echo ""
echo "Estimated training cost on GCP: \$35-53 (8-12 hours)"
