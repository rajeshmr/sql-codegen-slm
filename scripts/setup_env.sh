#!/bin/bash
# Setup script for sql-codegen conda environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_NAME="sql-codegen"

echo "🚀 Setting up sql-codegen environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Keeping existing environment. Exiting."
        exit 0
    fi
fi

# Create conda environment from environment.yml
echo "📦 Creating conda environment from environment.yml..."
conda env create -f "$PROJECT_DIR/environment.yml"

echo ""
echo "✅ Environment '${ENV_NAME}' created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Or use the helper script:"
echo "  source ./scripts/activate_env.sh"
echo ""
