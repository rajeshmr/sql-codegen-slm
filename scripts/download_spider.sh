#!/bin/bash
# Download Spider dataset for sql-codegen-slm project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SPIDER_DIR="$PROJECT_DIR/data/raw/spider"
DOWNLOAD_URL="https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ"
MANUAL_URL="https://yale-lily.github.io/spider"
ZIP_FILE="$SPIDER_DIR/spider.zip"

echo "🕷️  Spider Dataset Downloader"
echo "=============================="
echo ""

# Create directory if it doesn't exist
if [[ ! -d "$SPIDER_DIR" ]]; then
    echo "📁 Creating directory: $SPIDER_DIR"
    mkdir -p "$SPIDER_DIR"
fi

# Check if dataset already exists
if [[ -f "$SPIDER_DIR/train_spider.json" ]]; then
    echo "⚠️  Spider dataset already exists in $SPIDER_DIR"
    read -p "Do you want to re-download? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing dataset. Exiting."
        exit 0
    fi
    echo "🗑️  Removing existing dataset..."
    rm -rf "$SPIDER_DIR"/*
fi

# Check for required tools
if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
    echo "❌ Error: Neither curl nor wget is installed"
    exit 1
fi

if ! command -v unzip &> /dev/null; then
    echo "❌ Error: unzip is not installed"
    exit 1
fi

# Function to get file size
get_file_size() {
    if [[ -f "$1" ]]; then
        if [[ "$(uname)" == "Darwin" ]]; then
            stat -f%z "$1"
        else
            stat -c%s "$1"
        fi
    else
        echo "0"
    fi
}

# Function to format bytes
format_bytes() {
    local bytes=$1
    if [[ $bytes -ge 1073741824 ]]; then
        echo "$(echo "scale=2; $bytes/1073741824" | bc) GB"
    elif [[ $bytes -ge 1048576 ]]; then
        echo "$(echo "scale=2; $bytes/1048576" | bc) MB"
    elif [[ $bytes -ge 1024 ]]; then
        echo "$(echo "scale=2; $bytes/1024" | bc) KB"
    else
        echo "$bytes bytes"
    fi
}

echo "📥 Downloading Spider dataset..."
echo "   Source: Google Drive (Official Spider 1.0)"
echo ""

# Try downloading with gdown (if available) or curl
DOWNLOAD_SUCCESS=false

# Method 1: Try gdown (best for Google Drive)
if command -v gdown &> /dev/null; then
    echo "Using gdown for download..."
    if gdown "1TqleXec_OykOYFREKKtschzY29dUcVAQ" -O "$ZIP_FILE"; then
        DOWNLOAD_SUCCESS=true
    fi
fi

# Method 2: Try curl with Google Drive handling
if [[ "$DOWNLOAD_SUCCESS" == "false" ]] && command -v curl &> /dev/null; then
    echo "Using curl for download..."
    
    # Google Drive requires handling confirmation for large files
    CONFIRM=$(curl -sc /tmp/gcookie "${DOWNLOAD_URL}" | grep -o 'confirm=[^&]*' | head -1)
    
    if [[ -n "$CONFIRM" ]]; then
        curl -Lb /tmp/gcookie "${DOWNLOAD_URL}&${CONFIRM}" -o "$ZIP_FILE" --progress-bar
    else
        curl -L "${DOWNLOAD_URL}" -o "$ZIP_FILE" --progress-bar
    fi
    
    # Check if we got an HTML file instead of zip (Google Drive virus scan warning)
    if file "$ZIP_FILE" | grep -q "HTML"; then
        echo ""
        echo "⚠️  Google Drive returned an HTML page (virus scan warning)"
        rm -f "$ZIP_FILE"
    else
        DOWNLOAD_SUCCESS=true
    fi
fi

# Method 3: Try wget
if [[ "$DOWNLOAD_SUCCESS" == "false" ]] && command -v wget &> /dev/null; then
    echo "Using wget for download..."
    wget --no-check-certificate "${DOWNLOAD_URL}" -O "$ZIP_FILE" 2>&1 | tail -2
    
    if file "$ZIP_FILE" | grep -q "HTML"; then
        rm -f "$ZIP_FILE"
    else
        DOWNLOAD_SUCCESS=true
    fi
fi

# If download failed, provide manual instructions
if [[ "$DOWNLOAD_SUCCESS" == "false" ]] || [[ ! -f "$ZIP_FILE" ]] || [[ $(get_file_size "$ZIP_FILE") -lt 1000000 ]]; then
    echo ""
    echo "❌ Automatic download failed."
    echo ""
    echo "📋 MANUAL DOWNLOAD INSTRUCTIONS:"
    echo "================================="
    echo ""
    echo "1. Visit: $MANUAL_URL"
    echo ""
    echo "2. Click 'Download Spider dataset' or use this direct link:"
    echo "   https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ"
    echo ""
    echo "3. Save the file as 'spider.zip'"
    echo ""
    echo "4. Move the file to: $SPIDER_DIR/"
    echo "   mv ~/Downloads/spider.zip $SPIDER_DIR/"
    echo ""
    echo "5. Run this script again to extract and verify"
    echo ""
    
    # Check if user already has the zip file
    if [[ -f "$ZIP_FILE" ]]; then
        echo "Found existing spider.zip, attempting to extract..."
    else
        exit 1
    fi
fi

# Get download size
DOWNLOAD_SIZE=$(get_file_size "$ZIP_FILE")
echo ""
echo "✅ Download complete: $(format_bytes $DOWNLOAD_SIZE)"

# Extract the dataset
echo ""
echo "📦 Extracting dataset..."
cd "$SPIDER_DIR"
unzip -q -o "$ZIP_FILE"

# Spider dataset extracts to a 'spider' subdirectory, move contents up
if [[ -d "$SPIDER_DIR/spider" ]]; then
    # Move all contents including hidden files
    shopt -s dotglob
    mv "$SPIDER_DIR/spider"/* "$SPIDER_DIR/" 2>/dev/null || true
    shopt -u dotglob
    # Remove the now-empty spider directory
    rm -rf "$SPIDER_DIR/spider"
fi

# Remove zip file to save space
rm -f "$ZIP_FILE"
echo "✅ Extraction complete"

# Verify key files exist
echo ""
echo "🔍 Verifying dataset structure..."

ERRORS=0
WARNINGS=0

check_file() {
    if [[ -f "$SPIDER_DIR/$1" ]]; then
        local size=$(get_file_size "$SPIDER_DIR/$1")
        echo "✅ $1 ($(format_bytes $size))"
    else
        echo "❌ Missing: $1"
        ((ERRORS++))
    fi
}

check_dir() {
    if [[ -d "$SPIDER_DIR/$1" ]]; then
        local count=$(find "$SPIDER_DIR/$1" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
        echo "✅ $1/ ($count subdirectories)"
    else
        echo "❌ Missing directory: $1"
        ((ERRORS++))
    fi
}

check_file "train_spider.json"
check_file "train_others.json"
check_file "dev.json"
check_file "tables.json"
check_dir "database"

# Count databases and examples
echo ""
echo "📊 Dataset Statistics:"
echo "----------------------"

DB_COUNT=$(find "$SPIDER_DIR/database" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
echo "   Databases: $DB_COUNT"

# Count examples using Python (more reliable for JSON)
if command -v python3 &> /dev/null || command -v python &> /dev/null; then
    PYTHON_CMD=$(command -v python3 || command -v python)
    
    TRAIN_COUNT=$($PYTHON_CMD -c "import json; print(len(json.load(open('$SPIDER_DIR/train_spider.json'))))" 2>/dev/null || echo "N/A")
    TRAIN_OTHERS_COUNT=$($PYTHON_CMD -c "import json; print(len(json.load(open('$SPIDER_DIR/train_others.json'))))" 2>/dev/null || echo "N/A")
    DEV_COUNT=$($PYTHON_CMD -c "import json; print(len(json.load(open('$SPIDER_DIR/dev.json'))))" 2>/dev/null || echo "N/A")
    
    echo "   Training examples (train_spider.json): $TRAIN_COUNT"
    echo "   Training examples (train_others.json): $TRAIN_OTHERS_COUNT"
    echo "   Dev examples (dev.json): $DEV_COUNT"
    
    if [[ "$TRAIN_COUNT" != "N/A" && "$TRAIN_OTHERS_COUNT" != "N/A" ]]; then
        TOTAL_TRAIN=$((TRAIN_COUNT + TRAIN_OTHERS_COUNT))
        echo "   Total training examples: $TOTAL_TRAIN"
    fi
fi

# Create summary file
SUMMARY_FILE="$SPIDER_DIR/download_summary.txt"
echo ""
echo "📝 Creating download summary..."

cat > "$SUMMARY_FILE" << EOF
Spider Dataset Download Summary
===============================
Download Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Dataset Version: Spider 1.0
Source: Yale LILY Lab (https://yale-lily.github.io/spider)

File Counts:
- train_spider.json: $TRAIN_COUNT examples
- train_others.json: $TRAIN_OTHERS_COUNT examples  
- dev.json: $DEV_COUNT examples
- databases: $DB_COUNT directories

Verification Status:
- Errors: $ERRORS
- Warnings: $WARNINGS

Files Present:
$(ls -la "$SPIDER_DIR" | grep -v "^total" | grep -v "^d")

Database Directories: $DB_COUNT
EOF

if [[ $ERRORS -gt 0 ]]; then
    echo "⚠️  WARNINGS: Some expected files are missing" >> "$SUMMARY_FILE"
fi

echo "✅ Summary saved to: download_summary.txt"

# Final status
echo ""
echo "=========================================="
if [[ $ERRORS -eq 0 ]]; then
    echo "✅ Spider dataset downloaded successfully!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Verify with: python scripts/verify_spider.py"
    echo "  2. Explore data: cat data/raw/spider/train_spider.json | head"
    exit 0
else
    echo "❌ Download completed with $ERRORS error(s)"
    echo "=========================================="
    echo "Check the files manually and re-run if needed."
    exit 1
fi
