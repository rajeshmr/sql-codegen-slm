# SQL Code Generator - Fine-tuned Small Language Model

A fine-tuned Mistral-7B model that generates PostgreSQL queries from natural language, trained on the Spider dataset.

## Project Status

🚧 **In Progress - Module 0: Project Setup Complete**

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

## License

MIT
