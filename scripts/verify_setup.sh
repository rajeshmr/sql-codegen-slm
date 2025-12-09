#!/bin/bash
# Verify project setup for sql-codegen-slm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_NAME="sql-codegen"

echo "🔍 Verifying sql-codegen-slm project setup..."
echo ""

ERRORS=0

# Function to check if a directory exists
check_dir() {
    if [[ -d "$PROJECT_DIR/$1" ]]; then
        echo "✅ Directory exists: $1"
    else
        echo "❌ Missing directory: $1"
        ((ERRORS++))
    fi
}

# Function to check if a file exists
check_file() {
    if [[ -f "$PROJECT_DIR/$1" ]]; then
        echo "✅ File exists: $1"
    else
        echo "❌ Missing file: $1"
        ((ERRORS++))
    fi
}

# Function to check if a script is executable
check_executable() {
    if [[ -x "$PROJECT_DIR/$1" ]]; then
        echo "✅ Script is executable: $1"
    else
        echo "❌ Script not executable: $1"
        ((ERRORS++))
    fi
}

echo "📁 Checking directories..."
check_dir "data"
check_dir "data/raw"
check_dir "data/processed"
check_dir "data/demo"
check_dir "training"
check_dir "training/configs"
check_dir "training/logs"
check_dir "training/models"
check_dir "backend"
check_dir "backend/app"
check_dir "frontend"
check_dir "deployment"
check_dir "deployment/backend"
check_dir "deployment/frontend"
check_dir "tests"
check_dir "tests/data"
check_dir "tests/training"
check_dir "tests/backend"
check_dir "tests/integration"
check_dir "docs"
check_dir "scripts"

echo ""
echo "📄 Checking files..."
check_file ".gitignore"
check_file "README.md"
check_file "environment.yml"
check_file "requirements.txt"
check_file "setup.py"
check_file "data/__init__.py"
check_file "training/__init__.py"
check_file "backend/__init__.py"
check_file "tests/__init__.py"
check_file "docs/architecture.md"
check_file "docs/api.md"

echo ""
echo "🔧 Checking scripts..."
check_executable "scripts/setup_env.sh"
check_executable "scripts/activate_env.sh"
check_executable "scripts/clean_env.sh"
check_executable "scripts/verify_setup.sh"
check_executable "scripts/init_git.sh"

echo ""
echo "🐍 Checking conda environment..."
if command -v conda &> /dev/null; then
    echo "✅ Conda is installed"
    
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "✅ Conda environment '${ENV_NAME}' exists"
        
        # Check Python version in the environment
        eval "$(conda shell.bash hook)"
        conda activate "$ENV_NAME" 2>/dev/null
        
        if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
            PYTHON_VERSION=$(python --version 2>&1)
            if [[ "$PYTHON_VERSION" == *"3.10"* ]]; then
                echo "✅ Python version is 3.10: $PYTHON_VERSION"
            else
                echo "❌ Python version is not 3.10: $PYTHON_VERSION"
                ((ERRORS++))
            fi
            conda deactivate
        fi
    else
        echo "⚠️  Conda environment '${ENV_NAME}' not found (run ./scripts/setup_env.sh to create)"
    fi
else
    echo "⚠️  Conda is not installed (optional for verification)"
fi

echo ""
echo "=========================================="
if [[ $ERRORS -eq 0 ]]; then
    echo "✅ Project setup complete"
    echo "=========================================="
    exit 0
else
    echo "❌ Found $ERRORS error(s) in setup"
    echo "=========================================="
    exit 1
fi
