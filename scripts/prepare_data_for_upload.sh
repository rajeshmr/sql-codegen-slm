#!/bin/bash
# Upload Training Data to Google Cloud Storage
# SQL Codegen SLM - Module 2.1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/processed"

# GCS Configuration
BUCKET_NAME="${GCS_BUCKET:-sql-codegen-slm-data}"
GCS_DATA_PATH="gs://$BUCKET_NAME/data"

echo "� Upload Training Data to Google Cloud Storage"
echo "================================================"
echo ""

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo "❌ gsutil not found"
    echo ""
    echo "Install Google Cloud SDK:"
    echo "   https://cloud.google.com/sdk/docs/install"
    echo ""
    echo "Or on Mac: brew install google-cloud-sdk"
    exit 1
fi

# Check if data files exist
echo "🔍 Checking data files..."
MISSING_FILES=0

for file in train_postgres.jsonl val_postgres.jsonl test_postgres.jsonl; do
    if [[ -f "$DATA_DIR/$file" ]]; then
        SIZE=$(du -h "$DATA_DIR/$file" | cut -f1)
        LINES=$(wc -l < "$DATA_DIR/$file")
        echo "   ✅ $file: $LINES examples ($SIZE)"
    else
        echo "   ❌ $file: NOT FOUND"
        MISSING_FILES=1
    fi
done

if [[ $MISSING_FILES -eq 1 ]]; then
    echo ""
    echo "❌ Missing data files. Run the data pipeline first:"
    echo "   ./scripts/finalize_data.sh"
    exit 1
fi

echo ""

# Check GCS bucket
echo "☁️  Checking GCS bucket: $BUCKET_NAME"
if gsutil ls "gs://$BUCKET_NAME" &> /dev/null; then
    echo "   ✅ Bucket exists"
else
    echo "   📦 Creating bucket..."
    gsutil mb -l us "gs://$BUCKET_NAME"
    echo "   ✅ Bucket created"
fi

echo ""

# Upload to GCS
echo "📤 Uploading to GCS..."
echo "   Destination: $GCS_DATA_PATH/"
echo ""

gsutil -m cp \
    "$DATA_DIR/train_postgres.jsonl" \
    "$DATA_DIR/val_postgres.jsonl" \
    "$DATA_DIR/test_postgres.jsonl" \
    "$GCS_DATA_PATH/"

# Also upload metadata if exists
if [[ -f "$DATA_DIR/split_info.json" ]]; then
    gsutil cp "$DATA_DIR/split_info.json" "$GCS_DATA_PATH/"
fi
if [[ -f "$DATA_DIR/dataset_statistics.json" ]]; then
    gsutil cp "$DATA_DIR/dataset_statistics.json" "$GCS_DATA_PATH/"
fi

echo ""

# Verify upload
echo "🔍 Verifying upload..."
gsutil ls -l "$GCS_DATA_PATH/"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Upload Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Data location: $GCS_DATA_PATH/"
echo ""
echo "View in console:"
echo "   https://console.cloud.google.com/storage/browser/$BUCKET_NAME/data"
echo ""
echo "In Colab, download with:"
echo "   !gsutil -m cp $GCS_DATA_PATH/*.jsonl /content/data/"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 To use a different bucket, set GCS_BUCKET:"
echo "   export GCS_BUCKET=your-bucket-name"
echo "   ./scripts/prepare_data_for_upload.sh"
