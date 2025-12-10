# SQL Code Generator - Fine-tuned Small Language Model

A fine-tuned Mistral-7B model that generates PostgreSQL queries from natural language, trained on the Spider dataset.

## Project Status

🚧 **In Progress - Module 2.1: Training Environment Setup**

## Quick Start

Setup instructions coming soon.

## Project Structure

```
sql-codegen-slm/
├── data/                   # Dataset storage
│   ├── raw/               # Downloaded Spider dataset
│   ├── processed/         # Formatted training data
│   └── demo/              # Example schemas
├── training/              # Model training
│   ├── configs/           # Training configuration files
│   ├── logs/              # Training logs and metrics
│   └── models/            # Saved model checkpoints
├── backend/               # FastAPI application
│   └── app/               # API implementation
├── frontend/              # Next.js application
├── deployment/            # Deployment configurations
│   ├── backend/           # Dockerfile, Cloud Run configs
│   └── frontend/          # Frontend deployment configs
├── tests/                 # Test suites
│   ├── data/              # Data pipeline tests
│   ├── training/          # Training tests
│   ├── backend/           # API tests
│   └── integration/       # End-to-end tests
├── docs/                  # Documentation
│   ├── architecture.md    # System design
│   └── api.md             # API documentation
└── scripts/               # Helper scripts
    ├── setup_env.sh       # Create conda environment
    ├── activate_env.sh    # Activate environment
    ├── clean_env.sh       # Remove environment
    ├── verify_setup.sh    # Verify project setup
    └── init_git.sh        # Initialize git repository
```

## Technology Stack

- **Language**: Python 3.10
- **Model**: Mistral-7B (fine-tuned)
- **Backend**: FastAPI
- **Frontend**: Next.js
- **Training**: GCP (Google Cloud Platform)
- **Deployment**: GCP Cloud Run
- **Dataset**: Spider (Text-to-SQL)

## Environment Setup

```bash
# Create conda environment
./scripts/setup_env.sh

# Activate environment
source ./scripts/activate_env.sh

# Verify setup
./scripts/verify_setup.sh
```

## Data Download

The project uses the [Spider dataset](https://yale-lily.github.io/spider) - a large-scale text-to-SQL benchmark from Yale containing 10,181 questions across 200+ databases.

### Download Spider Dataset

```bash
# Download and extract Spider dataset
./scripts/download_spider.sh

# Verify the download
python scripts/verify_spider.py
```

### Expected Output

After successful download, `data/raw/spider/` will contain:

```
data/raw/spider/
├── train_spider.json          # ~8,659 training examples
├── train_others.json          # ~1,659 additional examples
├── dev.json                   # ~1,034 validation examples
├── tables.json                # Database schema definitions
├── database/                  # 200+ SQLite databases
│   ├── concert_singer/
│   │   └── concert_singer.sqlite
│   ├── pets_1/
│   │   └── pets_1.sqlite
│   └── ...
└── download_summary.txt       # Download verification
```

### Troubleshooting

**Download fails automatically:**
- Visit https://yale-lily.github.io/spider manually
- Download the Spider dataset zip file
- Place it in `data/raw/spider/spider.zip`
- Run `./scripts/download_spider.sh` again to extract

**JSON parsing errors:**
- Re-download the dataset (file may be corrupted)
- Check if extraction completed fully

**Reference:**
- [Spider Dataset Paper](https://arxiv.org/abs/1809.08887)
- [Spider Leaderboard](https://yale-lily.github.io/spider)

## Data Processing Pipeline

```
Spider Dataset (Yale NLP)
    │
    ▼
Module 1.1: Download
    │   ./scripts/download_spider.sh
    │   → data/raw/spider/
    │
    ▼
Module 1.2: Parse Schemas
    │   ./scripts/parse_schemas.sh
    │   → data/processed/schemas/
    │   → schema_index.json
    │
    ▼
Module 1.3: Format for Mistral
    │   ./scripts/convert_to_mistral.sh
    │   → data/processed/train_mistral.jsonl
    │   → data/processed/dev_mistral.jsonl
    │
    ▼
Module 1.4: Convert to PostgreSQL
    │   ./scripts/convert_to_postgres.sh
    │   → data/processed/train_postgres.jsonl
    │   → data/processed/dev_postgres.jsonl
    │
    ▼
Module 1.5: Create Splits + Demo Schemas  ✅ COMPLETE
    │   ./scripts/finalize_data.sh
    │   → data/processed/val_postgres.jsonl
    │   → data/processed/test_postgres.jsonl
    │   → data/demo/*.sql
    │
    ▼
READY FOR TRAINING 🚀
```

### Running the Pipeline

```bash
# Step 1: Download Spider dataset
./scripts/download_spider.sh

# Step 2: Parse and index schemas
./scripts/parse_schemas.sh

# Step 3: Convert to Mistral instruction format
./scripts/convert_to_mistral.sh

# Step 4: Convert SQLite to PostgreSQL syntax
./scripts/convert_to_postgres.sh

# Step 5: Create train/val/test splits and demo schemas
./scripts/finalize_data.sh

# Verify output
head -1 data/processed/train_postgres.jsonl | python -m json.tool
```

### Final Dataset

| Split | Examples | Description |
|-------|----------|-------------|
| Training | 6,016 | For model fine-tuning |
| Validation | 332 | For hyperparameter tuning |
| Test | 333 | For final evaluation |

## Demo Schemas

Five polished demo schemas for the web UI:

| Domain | Tables | Description |
|--------|--------|-------------|
| **E-commerce** | customers, products, orders, order_items, reviews | Online retail platform |
| **Finance** | customers, accounts, transactions, loans, credit_cards | Banking system |
| **Healthcare** | patients, doctors, departments, appointments, prescriptions | Hospital management |
| **SaaS** | organizations, users, plans, subscriptions, features | Multi-tenant platform |
| **Retail** | stores, employees, products, inventory, sales | Multi-store retail |

Each schema includes 10 sample questions ranging from simple to complex queries.

## Training Setup (Colab Pro+ with GCS)

### Prerequisites

1. Google Colab Pro+ subscription ($58.99/month)
2. GCP Project with Cloud Storage enabled
3. Google Cloud SDK installed locally

### Quick Start

1. Upload data to Google Cloud Storage:
   ```bash
   export GCS_BUCKET=your-bucket-name
   ./scripts/prepare_data_for_upload.sh
   ```

2. Open `notebooks/train_colab.ipynb` in Colab

3. Set your `PROJECT_ID` and `BUCKET_NAME` in the notebook

4. Run all cells in order

**Estimated time:** 8-12 hours on A100

### Cost Comparison

| Scenario | GCP VM | Colab Pro+ |
|----------|--------|------------|
| 1 run | $35-53 | $58.99/month |
| 5 runs | $175-265 | $58.99/month |
| Storage | ~$5/month | ~$0.20/month (GCS) |

### Advantages

- ✅ Cheaper for multiple experiments
- ✅ Fast data transfer with GCS
- ✅ Reliable checkpoint syncing
- ✅ TensorBoard integration
- ✅ Background execution (Pro+)

See [docs/training_colab_setup.md](docs/training_colab_setup.md) for detailed instructions.

## Training the Model

### Quick Start (Colab)

1. Open `notebooks/train_colab.ipynb` in Google Colab
2. Run all cells in order
3. Training takes 8-12 hours on A100
4. Model saves to GCS automatically

### Manual Training

```bash
python -m training.train \
  --config training/configs/mistral_lora_config.yaml
```

### Resume from Checkpoint

```bash
python -m training.train \
  --config training/configs/mistral_lora_config.yaml \
  --resume
```

### Monitoring Training

- **TensorBoard**: View in Colab or locally
  ```python
  %tensorboard --logdir /content/tensorboard
  ```
- **Logs**: Check `/content/logs/` for detailed logs
- **Checkpoints**: Every 500 steps synced to GCS

### Training Output

- **LoRA adapters**: ~100MB
- **Location**: `gs://sql-codegen-slm-data/models/`
- **Files**: `adapter_model.bin`, `adapter_config.json`, tokenizer

### Expected Results

| Metric | Expected Value |
|--------|----------------|
| Training loss | ~0.5-1.0 |
| Validation loss | ~1.0-1.5 |
| Training time | 8-12 hours (A100) |

## License

MIT
