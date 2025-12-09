#!/bin/bash
# Initialize git repository for sql-codegen-slm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🔧 Initializing git repository..."

cd "$PROJECT_DIR"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Error: git is not installed"
    exit 1
fi

# Check if already a git repository
if [[ -d ".git" ]]; then
    echo "✅ Git repository already initialized"
else
    echo "📦 Initializing new git repository..."
    git init
    echo "✅ Git repository initialized"
fi

# Check if there are any commits
if git rev-parse HEAD &> /dev/null; then
    echo "✅ Repository already has commits"
else
    echo "📝 Creating initial commit..."
    git add .
    git commit -m "feat: Initial project setup with conda env and directory structure - Module 0"
    echo "✅ Initial commit created"
fi

echo ""
echo "📊 Git status:"
git status

echo ""
echo "📜 Recent commits:"
git log --oneline -n 5 2>/dev/null || echo "No commits yet"
