#!/usr/bin/env python3
"""Create the Colab training notebook."""
import json
import os

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SQL Codegen SLM - Training Notebook\n",
                "\n",
                "Fine-tune Mistral-7B for PostgreSQL query generation.\n",
                "\n",
                "**Data:** `gs://sql-codegen-slm-data/data/`"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Check GPU"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!nvidia-smi\n",
                "\n",
                "import torch\n",
                "if torch.cuda.is_available():\n",
                "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
                "else:\n",
                "    print('No GPU!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Authenticate GCP"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "PROJECT_ID = 'your-gcp-project-id'\n",
                "BUCKET_NAME = 'sql-codegen-slm-data'\n",
                "\n",
                "import os\n",
                "os.environ['GCP_PROJECT_ID'] = PROJECT_ID\n",
                "os.environ['GCS_BUCKET'] = BUCKET_NAME\n",
                "\n",
                "from google.colab import auth\n",
                "auth.authenticate_user()\n",
                "!gcloud config set project {PROJECT_ID}\n",
                "print(f'Authenticated: {PROJECT_ID}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Clone Private Repo\n",
                "\n",
                "Get token at: https://github.com/settings/tokens (select repo scope)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from getpass import getpass\n",
                "import os\n",
                "\n",
                "GITHUB_TOKEN = getpass('GitHub Token: ')\n",
                "GITHUB_USER = 'your-github-username'\n",
                "REPO = 'sql-codegen-slm'\n",
                "\n",
                "if not os.path.exists(REPO):\n",
                "    !git clone https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{REPO}.git\n",
                "%cd {REPO}\n",
                "del GITHUB_TOKEN"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Install Dependencies"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q -r training/requirements.txt\n",
                "print('Dependencies installed')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Download Data from GCS"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!mkdir -p /content/data /content/models /content/logs /content/tensorboard\n",
                "!gsutil -m cp gs://{BUCKET_NAME}/data/*.jsonl /content/data/\n",
                "!wc -l /content/data/*.jsonl"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. Verify Environment"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from training.colab_setup import check_gpu, estimate_training_time\n",
                "check_gpu()\n",
                "estimate_training_time()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Start Training\n", "\n", "~8-12 hours on A100"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!python -m training.train --config training/configs/mistral_lora_config.yaml"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 8. Sync to GCS"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!gsutil -m rsync -r /content/models gs://{BUCKET_NAME}/models/\n",
                "print(f'Synced to gs://{BUCKET_NAME}/models/')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 9. TensorBoard"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%load_ext tensorboard\n",
                "%tensorboard --logdir /content/tensorboard"
            ]
        }
    ]
}

# Ensure notebooks directory exists
os.makedirs("notebooks", exist_ok=True)

# Write notebook
with open("notebooks/train_colab.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Created notebooks/train_colab.ipynb")

# Validate
with open("notebooks/train_colab.ipynb") as f:
    json.load(f)
print("Validated: JSON is valid")
