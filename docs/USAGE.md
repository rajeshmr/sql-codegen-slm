# Usage Guide

This guide covers how to use the fine-tuned Mistral-7B text-to-SQL model.

## Installation

### Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ GPU VRAM (with 4-bit quantization) or 32GB+ RAM (CPU)

### Install Dependencies

```bash
pip install transformers>=4.36.0 peft>=0.7.0 bitsandbytes>=0.41.0 accelerate>=0.25.0 torch>=2.0.0
```

## Loading the Model

### From HuggingFace Hub (Recommended)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configure 4-bit quantization for memory efficiency
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

print("Model loaded successfully!")
```

### CPU-Only Loading

For systems without a GPU:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model (no quantization for CPU)
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

# Load tokenizer and adapters
tokenizer = AutoTokenizer.from_pretrained("rajeshmanikka/mistral-7b-text-to-sql")
model = PeftModel.from_pretrained(base_model, "rajeshmanikka/mistral-7b-text-to-sql")
model.eval()
```

**Note**: CPU inference is significantly slower (30-60 seconds per query).

## Prompt Format

The model uses the Mistral instruction format. **Always use this exact template**:

```
<s>[INST] You are a SQL expert. Given the following PostgreSQL database schema, write a SQL query that answers the user's question.

Database Schema:
{schema}

Question: {question}

Generate only the SQL query without any explanation. [/INST]
```

### Template Components

| Component | Description |
|-----------|-------------|
| `<s>` | Start of sequence token |
| `[INST]` | Instruction start marker |
| `Database Schema:` | Section header for schema |
| `{schema}` | Your PostgreSQL CREATE TABLE statements |
| `Question:` | Section header for the question |
| `{question}` | Natural language question |
| `[/INST]` | Instruction end marker |

## Generating SQL

### Basic Generation Function

```python
def generate_sql(schema: str, question: str, max_tokens: int = 256) -> str:
    """Generate SQL query from schema and natural language question."""
    
    prompt = f"""<s>[INST] You are a SQL expert. Given the following PostgreSQL database schema, write a SQL query that answers the user's question.

Database Schema:
{schema}

Question: {question}

Generate only the SQL query without any explanation. [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract SQL after [/INST]
    if "[/INST]" in response:
        sql = response.split("[/INST]")[-1].strip()
    else:
        sql = response.strip()
    
    return sql
```

### Usage Example

```python
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

CREATE TABLE order_items (
    item_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_name VARCHAR(100),
    quantity INTEGER,
    price DECIMAL(10,2)
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

## Example Queries

### Simple SELECT

**Schema:**
```sql
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10,2),
    stock INTEGER
);
```

**Question:** "List all products in the Electronics category"

**Generated SQL:**
```sql
SELECT * FROM products WHERE category = 'Electronics';
```

### JOIN Query

**Schema:**
```sql
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    department_id INTEGER,
    salary DECIMAL(10,2)
);

CREATE TABLE departments (
    department_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    budget DECIMAL(15,2)
);
```

**Question:** "Show all employees with their department names"

**Generated SQL:**
```sql
SELECT e.name, d.name as department_name
FROM employees e
JOIN departments d ON e.department_id = d.department_id;
```

### Aggregation with GROUP BY

**Schema:**
```sql
CREATE TABLE sales (
    sale_id SERIAL PRIMARY KEY,
    product_id INTEGER,
    quantity INTEGER,
    sale_date DATE,
    amount DECIMAL(10,2)
);
```

**Question:** "What is the total sales amount for each month in 2024?"

**Generated SQL:**
```sql
SELECT DATE_TRUNC('month', sale_date) as month, SUM(amount) as total_sales
FROM sales
WHERE EXTRACT(YEAR FROM sale_date) = 2024
GROUP BY DATE_TRUNC('month', sale_date)
ORDER BY month;
```

### Subquery

**Schema:**
```sql
CREATE TABLE students (
    student_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    gpa DECIMAL(3,2)
);

CREATE TABLE enrollments (
    enrollment_id SERIAL PRIMARY KEY,
    student_id INTEGER REFERENCES students(student_id),
    course_id INTEGER
);
```

**Question:** "Find students who are enrolled in more than 3 courses"

**Generated SQL:**
```sql
SELECT s.name, COUNT(e.course_id) as course_count
FROM students s
JOIN enrollments e ON s.student_id = e.student_id
GROUP BY s.student_id, s.name
HAVING COUNT(e.course_id) > 3;
```

### Complex Multi-Table Query

**Schema:**
```sql
CREATE TABLE authors (author_id SERIAL PRIMARY KEY, name VARCHAR(100));
CREATE TABLE books (book_id SERIAL PRIMARY KEY, title VARCHAR(200), author_id INTEGER, published_year INTEGER);
CREATE TABLE reviews (review_id SERIAL PRIMARY KEY, book_id INTEGER, rating INTEGER, review_date DATE);
```

**Question:** "Find authors whose books have an average rating above 4, along with their average rating"

**Generated SQL:**
```sql
SELECT a.name, AVG(r.rating) as avg_rating
FROM authors a
JOIN books b ON a.author_id = b.author_id
JOIN reviews r ON b.book_id = r.book_id
GROUP BY a.author_id, a.name
HAVING AVG(r.rating) > 4
ORDER BY avg_rating DESC;
```

## Performance Tips

### Generation Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `temperature` | 0.1 | Lower = more deterministic |
| `max_new_tokens` | 256 | Sufficient for most queries |
| `top_p` | 0.95 | Nucleus sampling threshold |
| `do_sample` | True | Enable sampling |

### For Faster Inference

```python
# Use smaller max_new_tokens for simple queries
sql = generate_sql(schema, question, max_tokens=128)

# Disable sampling for deterministic output
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,  # Greedy decoding
    num_beams=1
)
```

### Batch Processing

```python
def generate_sql_batch(schemas: list, questions: list) -> list:
    """Generate SQL for multiple queries."""
    results = []
    
    for schema, question in zip(schemas, questions):
        sql = generate_sql(schema, question)
        results.append(sql)
    
    return results
```

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SQLRequest(BaseModel):
    schema: str
    question: str

class SQLResponse(BaseModel):
    sql: str

@app.post("/generate", response_model=SQLResponse)
async def generate_sql_endpoint(request: SQLRequest):
    sql = generate_sql(request.schema, request.question)
    return SQLResponse(sql=sql)
```

### Streamlit Integration

```python
import streamlit as st

st.title("Text-to-SQL Generator")

schema = st.text_area("Database Schema", height=200)
question = st.text_input("Question")

if st.button("Generate SQL"):
    if schema and question:
        sql = generate_sql(schema, question)
        st.code(sql, language="sql")
    else:
        st.warning("Please provide both schema and question")
```

### LangChain Integration

```python
from langchain.llms.base import LLM
from typing import Optional, List

class TextToSQLLLM(LLM):
    model: any = None
    tokenizer: any = None
    
    @property
    def _llm_type(self) -> str:
        return "text-to-sql"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Extract schema and question from prompt
        # Generate SQL using the model
        return generate_sql(schema, question)
```

## Best Practices

### Schema Formatting

1. **Use PostgreSQL syntax**: SERIAL, VARCHAR, REFERENCES
2. **Include all relevant tables**: The model needs complete context
3. **Add foreign key constraints**: Helps with JOIN generation
4. **Keep schemas concise**: Remove unnecessary columns for simple queries

### Question Formatting

1. **Be specific**: "Find customers who ordered in 2024" vs "Find customers"
2. **Use domain terms**: Match column/table names when possible
3. **Avoid ambiguity**: "total amount" vs "count of orders"

### Output Validation

Always validate generated SQL before execution:

```python
import sqlparse

def validate_sql(sql: str) -> bool:
    """Basic SQL validation."""
    try:
        parsed = sqlparse.parse(sql)
        return len(parsed) > 0 and parsed[0].get_type() != 'UNKNOWN'
    except:
        return False
```

## Limitations

1. **PostgreSQL Only**: Generates PostgreSQL syntax, not MySQL or SQLite
2. **Schema Required**: Cannot generate queries without schema context
3. **Complex Nesting**: Very deep subqueries may be inaccurate
4. **No Execution**: Does not validate against actual database
5. **Context Length**: Very large schemas (>50 tables) may exceed limits

## Troubleshooting

### Out of Memory

```python
# Use 8-bit quantization instead of 4-bit
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Or reduce max sequence length
inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
```

### Slow Generation

- Use GPU if available
- Reduce `max_new_tokens`
- Use greedy decoding (`do_sample=False`)

### Poor Quality Output

- Ensure schema is complete and correct
- Make question more specific
- Try rephrasing the question
- Check that prompt format matches template exactly
