#!/bin/bash
#
# Deploy Gradio app to HuggingFace Spaces
#
# This script:
# 1. Clones or creates the HF Spaces repository
# 2. Copies files from spaces/ directory
# 3. Commits and pushes to HuggingFace Spaces
#
# Usage:
#   bash scripts/deploy_to_spaces.sh
#
# Requirements:
#   - git installed
#   - HuggingFace CLI logged in (huggingface-cli login)
#   - Or HF_TOKEN environment variable set

set -e

# Configuration
HF_USERNAME="rajeshmanikka"
SPACE_NAME="text-to-sql-demo"
SPACE_REPO="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
SPACE_GIT_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

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

# Check HuggingFace authentication
echo ""
echo "🔐 Checking HuggingFace authentication..."

if [ -n "$HF_TOKEN" ]; then
    echo "   ✅ Using HF_TOKEN from environment"
    # Configure git to use token
    GIT_URL="https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
else
    # Check if logged in via CLI
    if command -v huggingface-cli &> /dev/null; then
        if huggingface-cli whoami &> /dev/null; then
            HF_USER=$(huggingface-cli whoami | head -1)
            echo "   ✅ Logged in as: ${HF_USER}"
            GIT_URL="${SPACE_GIT_URL}"
        else
            echo "   ⚠️  Not logged in to HuggingFace"
            echo "   Please run: huggingface-cli login"
            echo "   Or set HF_TOKEN environment variable"
            exit 1
        fi
    else
        echo "   ⚠️  huggingface-cli not found"
        echo "   Please install: pip install huggingface_hub"
        echo "   Or set HF_TOKEN environment variable"
        exit 1
    fi
fi

# Clean up any existing temp directory
if [ -d "$TEMP_DIR" ]; then
    echo ""
    echo "🧹 Cleaning up existing temp directory..."
    rm -rf "$TEMP_DIR"
fi

# Try to clone existing space or create new one
echo ""
echo "📥 Setting up Spaces repository..."

mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Try to clone existing repo
if git clone "$GIT_URL" . 2>/dev/null; then
    echo "   ✅ Cloned existing Space"
else
    echo "   📝 Creating new Space repository..."
    git init
    git remote add origin "$GIT_URL"
    
    # Create initial commit if needed
    echo "# ${SPACE_NAME}" > README.md
    git add README.md
    git commit -m "Initial commit" 2>/dev/null || true
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
