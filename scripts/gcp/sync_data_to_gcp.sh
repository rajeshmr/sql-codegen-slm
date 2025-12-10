#!/bin/bash
# Sync Data and Code to GCP Instance
# SQL Codegen SLM - Module 2.1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "📤 Syncing Data to GCP Instance"
echo "================================"
echo ""

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-YOUR_PROJECT_ID}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${GCP_INSTANCE_NAME:-sql-codegen-training}"
REMOTE_DIR="~/sql-codegen-slm"

# Check if project ID is set
if [[ "$PROJECT_ID" == "YOUR_PROJECT_ID" ]]; then
    echo "❌ GCP Project ID not set"
    echo "   export GCP_PROJECT_ID=your-project-id"
    exit 1
fi

# Check if instance exists and is running
echo "🔍 Checking instance status..."
INSTANCE_STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --format="get(status)" 2>/dev/null || echo "NOT_FOUND")

if [[ "$INSTANCE_STATUS" == "NOT_FOUND" ]]; then
    echo "❌ Instance '$INSTANCE_NAME' not found"
    echo "   Run: ./scripts/gcp/create_training_instance.sh"
    exit 1
fi

if [[ "$INSTANCE_STATUS" != "RUNNING" ]]; then
    echo "⚠️  Instance is $INSTANCE_STATUS"
    echo "   Starting instance..."
    gcloud compute instances start "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE"
    echo "   Waiting for instance to start..."
    sleep 30
fi

echo "✅ Instance is running"
echo ""

# Create remote directory
echo "📁 Creating remote directory..."
gcloud compute ssh "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --command="mkdir -p $REMOTE_DIR"

# Sync code files (excluding large files and data)
echo ""
echo "📦 Syncing code files..."
gcloud compute scp --recurse \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --compress \
    "$PROJECT_DIR/training" \
    "$PROJECT_DIR/data/__init__.py" \
    "$PROJECT_DIR/data/split_creator.py" \
    "$PROJECT_DIR/data/dataset_stats.py" \
    "$PROJECT_DIR/requirements.txt" \
    "$PROJECT_DIR/setup.py" \
    "$INSTANCE_NAME:$REMOTE_DIR/"

# Sync processed data files
echo ""
echo "📊 Syncing training data..."
gcloud compute ssh "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --command="mkdir -p $REMOTE_DIR/data/processed"

# Sync each data file with progress
for file in train_postgres.jsonl val_postgres.jsonl test_postgres.jsonl split_info.json dataset_statistics.json; do
    if [[ -f "$PROJECT_DIR/data/processed/$file" ]]; then
        echo "   Syncing $file..."
        gcloud compute scp \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --compress \
            "$PROJECT_DIR/data/processed/$file" \
            "$INSTANCE_NAME:$REMOTE_DIR/data/processed/"
    fi
done

# Verify transfer
echo ""
echo "🔍 Verifying transfer..."
gcloud compute ssh "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --command="ls -la $REMOTE_DIR/data/processed/"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Data Sync Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Connect: ./scripts/gcp/connect_to_instance.sh"
echo "  2. On instance, run:"
echo "     cd $REMOTE_DIR"
echo "     pip install -r training/requirements.txt"
echo "     python -m training.train"
