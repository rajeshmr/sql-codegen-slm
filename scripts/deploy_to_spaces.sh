#!/bin/bash
#
# Deploy Gradio app to HuggingFace Spaces
#
# This script:
# 1. Creates the HF Space if it doesn't exist (using huggingface_hub)
# 2. Clones the Spaces repository
# 3. Copies files from spaces/ directory
# 4. Commits and pushes to HuggingFace Spaces
#
# Usage:
#   bash scripts/deploy_to_spaces.sh
#
# Requirements:
#   - git installed
#   - huggingface_hub installed (pip install huggingface_hub)
#   - HuggingFace CLI logged in (huggingface-cli login) or HF_TOKEN set

set -e

# Configuration
HF_USERNAME="rajeshmanikka"
SPACE_NAME="text-to-sql-demo"
SPACE_ID="${HF_USERNAME}/${SPACE_NAME}"
SPACE_REPO="https://huggingface.co/spaces/${SPACE_ID}"

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SPACES_DIR="${PROJECT_ROOT}/spaces"
TEMP_DIR="${PROJECT_ROOT}/.spaces_deploy_temp"

echo "============================================================"
echo "HuggingFace Spaces Deployment"
echo "============================================================"
echo "Space: ${SPACE_REPO}"
echo "Source: ${SPACES_DIR}"
echo "============================================================"

# Check if spaces directory exists
if [ ! -d "$SPACES_DIR" ]; then
    echo "❌ Error: spaces/ directory not found at ${SPACES_DIR}"
    exit 1
fi

# Check for required files
echo ""
echo "🔍 Checking required files..."
for file in "app.py" "requirements.txt" "README.md"; do
    if [ -f "${SPACES_DIR}/${file}" ]; then
        echo "   ✅ ${file}"
    else
        echo "   ❌ ${file} not found"
        exit 1
    fi
done

# Check HuggingFace authentication and get token
echo ""
echo "🔐 Checking HuggingFace authentication..."

if [ -n "$HF_TOKEN" ]; then
    echo "   ✅ Using HF_TOKEN from environment"
    TOKEN="$HF_TOKEN"
else
    # Try to get token from huggingface-cli
    if command -v huggingface-cli &> /dev/null; then
        # Check if logged in
        if huggingface-cli whoami &> /dev/null 2>&1 || hf whoami &> /dev/null 2>&1; then
            echo "   ✅ Logged in to HuggingFace"
            # Try to get token from the token file
            TOKEN_FILE="$HOME/.cache/huggingface/token"
            if [ -f "$TOKEN_FILE" ]; then
                TOKEN=$(cat "$TOKEN_FILE")
                echo "   ✅ Found cached token"
            else
                echo "   ⚠️  Could not find cached token"
                echo "   Please set HF_TOKEN environment variable"
                exit 1
            fi
        else
            echo "   ⚠️  Not logged in to HuggingFace"
            echo "   Please run: huggingface-cli login"
            echo "   Or set HF_TOKEN environment variable"
            exit 1
        fi
    else
        echo "   ⚠️  huggingface-cli not found"
        echo "   Please install: pip install huggingface_hub"
        exit 1
    fi
fi

# Create the Space if it doesn't exist
echo ""
echo "📦 Creating Space repository (if needed)..."

python3 << EOF
from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
token = "${TOKEN}"

try:
    # Try to create the repo (will succeed if it doesn't exist)
    create_repo(
        repo_id="${SPACE_ID}",
        repo_type="space",
        space_sdk="gradio",
        token=token,
        exist_ok=True,
        private=False
    )
    print("   ✅ Space repository ready")
except Exception as e:
    print(f"   ⚠️  Note: {e}")
    print("   Continuing anyway...")
EOF

# Clean up any existing temp directory
if [ -d "$TEMP_DIR" ]; then
    echo ""
    echo "🧹 Cleaning up existing temp directory..."
    rm -rf "$TEMP_DIR"
fi

# Clone the space
echo ""
echo "📥 Cloning Spaces repository..."

mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Build git URL with token
GIT_URL="https://${HF_USERNAME}:${TOKEN}@huggingface.co/spaces/${SPACE_ID}"

# Clone the repo
if git clone "$GIT_URL" . 2>/dev/null; then
    echo "   ✅ Cloned Space"
else
    echo "   📝 Initializing new repository..."
    git init
    git remote add origin "$GIT_URL"
fi

# Copy files from spaces/ directory
echo ""
echo "📋 Copying files..."
cp -v "${SPACES_DIR}/app.py" .
cp -v "${SPACES_DIR}/requirements.txt" .
cp -v "${SPACES_DIR}/README.md" .

# Copy any additional files if they exist
for file in ".gitattributes" "style.css" "config.json"; do
    if [ -f "${SPACES_DIR}/${file}" ]; then
        cp -v "${SPACES_DIR}/${file}" .
    fi
done

# Stage all changes
echo ""
echo "📦 Staging changes..."
git add -A

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "   ℹ️  No changes to commit"
else
    # Commit changes
    echo ""
    echo "💾 Committing changes..."
    git commit -m "Deploy text-to-SQL demo app

- Gradio interface for SQL generation
- Loads model from HuggingFace Hub
- Includes example queries"

    # Push to HuggingFace Spaces
    echo ""
    echo "🚀 Pushing to HuggingFace Spaces..."
    
    # Set up git credentials if using token
    if [ -n "$HF_TOKEN" ]; then
        git push "$GIT_URL" main 2>&1 || git push "$GIT_URL" master 2>&1
    else
        git push origin main 2>&1 || git push origin master 2>&1
    fi
    
    echo "   ✅ Pushed successfully!"
fi

# Clean up
echo ""
echo "🧹 Cleaning up..."
cd "$PROJECT_ROOT"
rm -rf "$TEMP_DIR"

# Final summary
echo ""
echo "============================================================"
echo "✅ DEPLOYMENT COMPLETE"
echo "============================================================"
echo ""
echo "Your Space is now deploying at:"
echo "   ${SPACE_REPO}"
echo ""
echo "Note: It may take a few minutes for the Space to build."
echo "      Check the 'Logs' tab on HuggingFace for build status."
echo ""
echo "============================================================"
