# SQL Code Generator - Fine-tuned Mistral-7B for Text-to-SQL

[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-mistral--7b--text--to--sql-blue)](https://huggingface.co/rajeshmanikka/mistral-7b-text-to-sql)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Demo-text--to--sql--demo-orange)](https://huggingface.co/spaces/rajeshmanikka/text-to-sql-demo)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)

A fine-tuned Mistral-7B model that generates PostgreSQL queries from natural language, trained on the Spider dataset using LoRA (Low-Rank Adaptation).

## 🎯 Quick Links

| Resource | Link |
|----------|------|
| **🚀 Live Demo** | [HuggingFace Spaces](https://huggingface.co/spaces/rajeshmanikka/text-to-sql-demo) |
| **🤗 Model** | [rajeshmanikka/mistral-7b-text-to-sql](https://huggingface.co/rajeshmanikka/mistral-7b-text-to-sql) |
| **📊 Dataset** | [Spider (Yale)](https://yale-lily.github.io/spider) |
| **📖 Usage Guide** | [docs/USAGE.md](docs/USAGE.md) |

## ✨ Key Results

| Metric | Value |
|--------|-------|
| **Training Time** | 8h 55m (A100 40GB) |
| **Training Cost** | ~$60 (Colab Pro+) |
| **Final Train Loss** | 0.043 |
| **Final Val Loss** | 1.085 |
| **Trainable Params** | 42M (1.11%) |
| **Model Size** | ~164MB (LoRA adapters) |

## 🚀 Quick Start

### Use the Model (from HuggingFace Hub)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

# Load tokenizer and LoRA adapters
tokenizer = AutoTokenizer.from_pretrained("rajeshmanikka/mistral-7b-text-to-sql")
model = PeftModel.from_pretrained(base_model, "rajeshmanikka/mistral-7b-text-to-sql")

# Generate SQL
schema = """
CREATE TABLE customers (customer_id SERIAL PRIMARY KEY, name VARCHAR(100));
CREATE TABLE orders (order_id SERIAL PRIMARY KEY, customer_id INTEGER, total DECIMAL);
"""

prompt = f"""<s>[INST] You are a SQL expert. Given the following PostgreSQL database schema, write a SQL query that answers the user's question.

Database Schema:
{schema}

Question: Find the top 5 customers by total order amount

Generate only the SQL query without any explanation. [/INST]"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
sql = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
print(sql)
```

### Example Output

```sql
SELECT c.name, SUM(o.total) as total_amount
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
ORDER BY total_amount DESC
LIMIT 5;
```

## 📁 Project Structure

```
sql-codegen-slm/
├── data/                   # Dataset storage
│   ├── raw/               # Downloaded Spider dataset
│   ├── processed/         # Formatted training data
│   └── demo/              # Example schemas
├── training/              # Model training
│   ├── configs/           # Training configuration files
│   ├── train.py           # Main training script
│   └── validation.py      # Validation utilities
├── notebooks/             # Jupyter notebooks
│   ├── train_colab.ipynb  # Training notebook
│   └── validation_notebook.ipynb
├── spaces/                # HuggingFace Spaces demo
│   ├── app.py             # Gradio application
│   └── requirements.txt
├── scripts/               # Utility scripts
│   ├── upload_to_hf.py    # Upload model to HF Hub
│   └── deploy_to_spaces.sh # Deploy demo to Spaces
├── docs/                  # Documentation
│   ├── MODEL_CARD.md      # HuggingFace model card
│   ├── USAGE.md           # Usage guide
│   └── DEPLOYMENT.md      # Deployment instructions
└── tests/                 # Test suites
```

## 🛠️ Technology Stack

- **Language**: Python 3.10
- **Base Model**: Mistral-7B-v0.1
- **Fine-tuning**: LoRA (PEFT)
- **Quantization**: 4-bit NF4 (bitsandbytes)
- **Training**: Google Colab Pro+ (A100)
- **Dataset**: Spider (Text-to-SQL)
- **Demo**: Gradio on HuggingFace Spaces

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

## Pre-Training Validation

**IMPORTANT:** Run validation before full training to catch issues early.

### Quick Validation (10 minutes)

1. Open `notebooks/validation_notebook.ipynb` in Colab
2. Run all cells
3. Verify all tests pass ✅

### What Gets Validated

- ✅ Data loading (20 examples)
- ✅ Model initialization (4-bit + LoRA)
- ✅ Training loop (20 steps)
- ✅ Checkpoint saving to GCS
- ✅ Checkpoint loading
- ✅ Model inference (SQL generation)
- ✅ GPU memory usage (<20GB)

### Expected Validation Results

- All tests pass ✅
- Training loss decreases (3.5 → 1.5 in 20 steps)
- Model generates valid SQL
- Checkpoints save to GCS
- Memory usage <20GB

### If Validation Fails

1. Check error messages carefully
2. Verify GCS authentication
3. Check GPU allocation (need A100 or V100)
4. Review training/logs/ for details
5. Ask for help with specific error

### After Validation Passes

✅ Ready for full training! Proceed to training section below.

## Training the Model

### Quick Start (Colab)

1. **First**: Run validation notebook (see above)
2. Open `notebooks/train_colab.ipynb` in Google Colab
3. Run all cells in order
4. Training takes 8-12 hours on A100
5. Model saves to GCS automatically

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

### Training Results

| Metric | Value |
|--------|-------|
| Final Training Loss | 0.043 |
| Final Validation Loss | 1.085 |
| Training Time | 8h 55m |
| Total Steps | 1,128 |
| GPU Memory Used | 13.05 GB |

### Loss Progression

```
Epoch 0.1: 0.73 → Epoch 0.5: 0.03 → Epoch 1.0: 0.01 → Epoch 2.0: 0.008 → Epoch 3.0: 0.007
```

## 🧪 Example Generated SQL

| Question | Generated SQL |
|----------|---------------|
| Find top 5 customers by total orders | `SELECT c.name, SUM(o.total) FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id ORDER BY SUM(o.total) DESC LIMIT 5` |
| Average price in Electronics category | `SELECT AVG(price) FROM products WHERE category = 'Electronics'` |
| Employees with department budgets | `SELECT e.name, d.budget FROM employees e JOIN departments d ON e.department_id = d.department_id` |

## 📚 Methodology

### Why LoRA?
- **Efficiency**: Only 1.11% of parameters trained (42M vs 3.8B)
- **Memory**: Fits on consumer GPUs with 4-bit quantization
- **Speed**: 9 hours vs days for full fine-tuning
- **Quality**: Comparable results to full fine-tuning for this task

### Why Spider Dataset?
- **Diversity**: 200+ database schemas across 138 domains
- **Complexity**: Includes JOINs, aggregations, subqueries, nested queries
- **Quality**: Human-annotated by Yale researchers
- **Benchmark**: Standard evaluation for text-to-SQL models

### Why Colab Pro+?
- **Cost**: $60/month vs $200+ for equivalent GCP compute
- **Simplicity**: No infrastructure setup required
- **A100 Access**: High-priority GPU allocation
- **Checkpointing**: Automatic saves to Google Cloud Storage

## ⚠️ Limitations

1. **PostgreSQL Only**: Generates PostgreSQL syntax (not MySQL/SQLite)
2. **Schema Required**: Must provide complete database schema
3. **Complex Queries**: Very deep nesting may be inaccurate
4. **No Validation**: Generated SQL should be reviewed before execution

## 📖 References

### Spider Dataset
```bibtex
@inproceedings{yu2018spider,
  title={Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  booktitle={EMNLP},
  year={2018}
}
```

### LoRA
```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```

### Mistral
```bibtex
@article{jiang2023mistral,
  title={Mistral 7B},
  author={Jiang, Albert Q and Sablayrolles, Alexandre and Mensch, Arthur and Bamford, Chris and Chaplot, Devendra Singh and Casas, Diego de las and Bressand, Florian and Lengyel, Gianna and Lample, Guillaume and Saulnier, Lucile and others},
  journal={arXiv preprint arXiv:2310.06825},
  year={2023}
}
```

## 🙏 Acknowledgments

- **[Mistral AI](https://mistral.ai/)** for the Mistral-7B base model
- **[Yale LILY Lab](https://yale-lily.github.io/)** for the Spider dataset
- **[Hugging Face](https://huggingface.co/)** for transformers and PEFT libraries
- **[Tim Dettmers](https://github.com/TimDettmers)** for bitsandbytes quantization

## 📬 Contact

- **GitHub**: [rajeshmr/sql-codegen-slm](https://github.com/rajeshmr/sql-codegen-slm)
- **HuggingFace**: [rajeshmanikka](https://huggingface.co/rajeshmanikka)

## License

Apache 2.0
