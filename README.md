# SQL Code Generator - Fine-tuned Small Language Model

A fine-tuned Mistral-7B model that generates PostgreSQL queries from natural language, trained on the Spider dataset.

## Project Status

🚧 **In Progress - Module 1.1: Spider Dataset Download**

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

## License

MIT
