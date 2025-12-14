---
language: en
license: apache-2.0
base_model: mistralai/Mistral-7B-v0.1
tags:
  - text-to-sql
  - sql-generation
  - mistral
  - lora
  - peft
  - postgresql
  - nlp
  - code-generation
datasets:
  - spider
library_name: peft
pipeline_tag: text-generation
model-index:
  - name: mistral-7b-text-to-sql
    results:
      - task:
          type: text-to-sql
        dataset:
          name: Spider
          type: spider
        metrics:
          - name: Training Loss
            type: loss
            value: 0.043
          - name: Validation Loss
            type: loss
            value: 1.085
---

# Mistral-7B Text-to-SQL

A fine-tuned [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) model for generating PostgreSQL queries from natural language questions. This model uses LoRA (Low-Rank Adaptation) for efficient fine-tuning on the Spider text-to-SQL benchmark dataset.

## Model Description

This model converts natural language questions into PostgreSQL queries given a database schema. It was trained using parameter-efficient fine-tuning (LoRA) on the Spider dataset, which contains complex SQL queries across 200+ database domains.

### Key Features

- **PostgreSQL Syntax**: Generates PostgreSQL-compatible SQL (not SQLite)
- **Schema-Aware**: Takes database schema as context for accurate query generation
- **Complex Queries**: Supports JOINs, aggregations, subqueries, GROUP BY, ORDER BY, and more
- **Efficient**: LoRA adapters are only ~164MB (vs 14GB for full model)
- **4-bit Quantization**: Can run on consumer GPUs with 8GB+ VRAM

### Supported SQL Operations

| Operation | Support |
|-----------|---------|
| SELECT with WHERE | ✅ |
| JOINs (INNER, LEFT, RIGHT) | ✅ |
| GROUP BY with aggregations | ✅ |
| ORDER BY with LIMIT | ✅ |
| Subqueries | ✅ |
| HAVING clauses | ✅ |
| DISTINCT | ✅ |
| UNION | ✅ |

## Training Details

### Dataset

- **Name**: [Spider](https://yale-lily.github.io/spider) (Yale Semantic Parsing and Text-to-SQL Challenge)
- **Training Examples**: 8,234
- **Validation Examples**: 495
- **Test Examples**: 494
- **Databases**: 200+ diverse schemas (e-commerce, healthcare, finance, etc.)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | mistralai/Mistral-7B-v0.1 |
| Method | LoRA (Low-Rank Adaptation) |
| Hardware | NVIDIA A100 40GB (Google Colab Pro+) |
| Training Time | 8 hours 55 minutes |
| Epochs | 3 |
| Batch Size | 4 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 16 |
| Learning Rate | 2e-4 |
| Optimizer | paged_adamw_32bit |
| LR Scheduler | Cosine with warmup (3%) |
| Precision | bfloat16 |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable Parameters | 41,943,040 (1.11%) |
| Total Parameters | 3,794,014,208 |

### Quantization (4-bit)

| Parameter | Value |
|-----------|-------|
| Quantization | 4-bit NormalFloat (NF4) |
| Compute Dtype | float16 |
| Double Quantization | Enabled |

## Training Results

### Loss Progression

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 0.5 | 0.027 | - |
| 1.0 | 0.013 | - |
| 1.33 | 0.010 | 1.085 |
| 2.0 | 0.008 | - |
| 2.66 | 0.007 | 1.219 |
| 3.0 | 0.043 (avg) | 1.085 |

### Final Metrics

- **Final Training Loss**: 0.043
- **Final Validation Loss**: 1.085
- **Training Samples/Second**: 0.561
- **Total Training Steps**: 1,128
- **GPU Memory Used**: 13.05 GB (peak)

## Usage

### Installation

```bash
pip install transformers peft bitsandbytes accelerate torch
```

### Loading the Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("rajeshmanikka/mistral-7b-text-to-sql")
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "rajeshmanikka/mistral-7b-text-to-sql")
model.eval()
```

### Inference Example

```python
def generate_sql(schema: str, question: str) -> str:
    """Generate SQL query from natural language question."""
    
    prompt = f"""<s>[INST] You are a SQL expert. Given the following PostgreSQL database schema, write a SQL query that answers the user's question.

Database Schema:
{schema}

Question: {question}

Generate only the SQL query without any explanation. [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract SQL after the prompt
    sql = response.split("[/INST]")[-1].strip()
    return sql

# Example usage
schema = """
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    created_at TIMESTAMP
);

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    total DECIMAL(10,2),
    order_date DATE
);
"""

question = "Find the top 5 customers by total order amount"

sql = generate_sql(schema, question)
print(sql)
```

### Expected Output

```sql
SELECT c.name, SUM(o.total) as total_amount
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
ORDER BY total_amount DESC
LIMIT 5;
```

## Prompt Format

This model uses the Mistral instruction format:

```
<s>[INST] You are a SQL expert. Given the following PostgreSQL database schema, write a SQL query that answers the user's question.

Database Schema:
{schema}

Question: {question}

Generate only the SQL query without any explanation. [/INST]
```

### Important Notes

- Always include the full database schema in the prompt
- Use PostgreSQL-style syntax in your schema (SERIAL, VARCHAR, etc.)
- The model generates only the SQL query, not explanations
- For best results, keep questions clear and specific

## Limitations

1. **PostgreSQL Only**: The model is trained on PostgreSQL syntax. SQLite or MySQL queries may have syntax differences.

2. **Schema Required**: The model requires the database schema as context. It cannot generate queries without knowing the table structure.

3. **Complex Nested Queries**: While the model handles most SQL operations, extremely complex nested subqueries may not always be accurate.

4. **Domain-Specific Terms**: The model works best with common database domains. Highly specialized terminology may require clearer questions.

5. **No Query Validation**: The model generates SQL based on patterns learned during training. Always validate generated queries before execution.

6. **Context Length**: Very large schemas (>50 tables) may exceed the model's context window.

## Ethical Considerations

- **SQL Injection**: Generated queries should be parameterized before use in production applications
- **Data Privacy**: Do not include sensitive data in prompts
- **Validation**: Always review generated SQL before executing on production databases

## Dataset

This model was trained on the [Spider dataset](https://yale-lily.github.io/spider):

> Spider is a large-scale complex and cross-domain semantic parsing and text-to-SQL dataset annotated by 11 Yale students. It consists of 10,181 questions and 5,693 unique complex SQL queries on 200 databases with multiple tables covering 138 different domains.

### Citation (Spider Dataset)

```bibtex
@inproceedings{yu2018spider,
  title={Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={3911--3921},
  year={2018}
}
```

## Repository

- **GitHub**: [https://github.com/rajeshmanikka/sql-codegen-slm](https://github.com/rajeshmanikka/sql-codegen-slm)
- **Training Code**: Available in the repository under `training/`
- **Data Processing**: Scripts for Spider dataset processing included

## Citation

If you use this model, please cite:

```bibtex
@misc{manikka2024mistral7btexttosql,
  title={Mistral-7B Text-to-SQL: A LoRA Fine-tuned Model for PostgreSQL Query Generation},
  author={Rajesh Manikka},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/rajeshmanikka/mistral-7b-text-to-sql}
}
```

## Acknowledgments

- **[Mistral AI](https://mistral.ai/)** for the Mistral-7B base model
- **[Yale LILY Lab](https://yale-lily.github.io/)** for the Spider dataset
- **[Hugging Face](https://huggingface.co/)** for the transformers and PEFT libraries
- **[Tim Dettmers](https://github.com/TimDettmers)** for bitsandbytes quantization

## License

This model is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

The base model (Mistral-7B) is also released under Apache 2.0. The Spider dataset is released for research purposes under its own license.

---

**Model Card Version**: 1.0  
**Last Updated**: December 2024  
**Contact**: [GitHub Issues](https://github.com/rajeshmanikka/sql-codegen-slm/issues)
