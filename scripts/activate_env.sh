#!/bin/bash
# Activate sql-codegen conda environment

ENV_NAME="sql-codegen"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    exit 1
fi

# Source conda for shell integration
eval "$(conda shell.bash hook)"

# Check if environment exists
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "❌ Error: Environment '${ENV_NAME}' does not exist."
    echo "Run ./scripts/setup_env.sh first to create it."
    exit 1
fi

# Activate the environment
conda activate "$ENV_NAME"

echo "✅ Activated environment: $CONDA_DEFAULT_ENV"
echo "🐍 Python version: $(python --version)"
echo "📍 Python path: $(which python)"
