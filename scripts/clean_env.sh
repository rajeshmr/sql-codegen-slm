#!/bin/bash
# Remove sql-codegen conda environment

ENV_NAME="sql-codegen"

echo "🧹 Cleaning up sql-codegen environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    exit 1
fi

# Source conda for shell integration
eval "$(conda shell.bash hook)"

# Deactivate if currently in the environment
if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
    echo "📤 Deactivating current environment..."
    conda deactivate
fi

# Check if environment exists
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' does not exist. Nothing to remove."
    exit 0
fi

# Remove the environment
echo "🗑️  Removing environment '${ENV_NAME}'..."
conda env remove -n "$ENV_NAME" -y

echo ""
echo "✅ Environment '${ENV_NAME}' has been removed."
