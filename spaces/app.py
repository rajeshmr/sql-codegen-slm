"""
Gradio demo for Mistral-7B Text-to-SQL model.

This app loads the fine-tuned LoRA adapters from HuggingFace Hub
and generates PostgreSQL queries from natural language questions.
"""

import spaces
import gradio as gr
import torch
import sqlparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Model configuration
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_MODEL = "rajeshmanikka/mistral-7b-text-to-sql"

# Global model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the model with 4-bit quantization and LoRA adapters."""
    global model, tokenizer
    
    if model is not None:
        return  # Already loaded
    
    print("Loading model... This may take a few minutes.")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA adapters
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL)
    model.eval()
    
    print("Model loaded successfully!")


def _generate_sql_impl(schema: str, question: str) -> str:
    """Internal implementation of SQL generation."""
    try:
        if not schema.strip():
            return "Error: Please provide a database schema."
        
        if not question.strip():
            return "Error: Please provide a question."
        
        # Ensure model is loaded
        print(f"[DEBUG] Starting generation...")
        print(f"[DEBUG] Schema length: {len(schema)}, Question: {question[:50]}...")
        load_model()
        
        if model is None:
            return "Error: Model failed to load. Check logs for details."
        
        if tokenizer is None:
            return "Error: Tokenizer failed to load. Check logs for details."
        
        # Format prompt using Mistral instruction template
        prompt = f"""<s>[INST] You are a SQL expert. Given the following PostgreSQL database schema, write a SQL query that answers the user's question.

Database Schema:
{schema.strip()}

Question: {question.strip()}

Generate only the SQL query without any explanation. [/INST]"""

        print(f"[DEBUG] Prompt length: {len(prompt)} chars")
        
        # Tokenize
        print(f"[DEBUG] Tokenizing...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        print(f"[DEBUG] Input tokens: {inputs['input_ids'].shape}")
        
        # Move to device
        print(f"[DEBUG] Model device: {model.device}")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        print(f"[DEBUG] Generating...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        print(f"[DEBUG] Output tokens: {outputs.shape}")
        
        # Decode and extract SQL
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[DEBUG] Response length: {len(response)} chars")
        
        # Extract SQL after [/INST]
        if "[/INST]" in response:
            sql = response.split("[/INST]")[-1].strip()
        else:
            sql = response.strip()
        
        # Clean up any trailing content
        sql = sql.split("</s>")[0].strip()
        sql = sql.split("[INST]")[0].strip()
        
        # Format SQL for better readability
        sql = sqlparse.format(
            sql,
            reindent=True,
            keyword_case='upper',
            indent_width=2
        )
        
        print(f"[DEBUG] Generated SQL: {sql[:100]}...")
        return sql
        
    except Exception as e:
        import traceback
        error_msg = f"Error during generation: {str(e)}"
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        return f"Error: {str(e)}\n\nPlease check the logs for more details."


@spaces.GPU(duration=120)
def generate_sql(schema: str, question: str) -> str:
    """Generate SQL query from schema and natural language question."""
    return _generate_sql_impl(schema, question)


# Example queries for the demo
EXAMPLES = [
    [
        """CREATE TABLE customers (
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
);""",
        "Find the top 5 customers by total order amount"
    ],
    [
        """CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    department_id INTEGER,
    salary DECIMAL(10,2),
    hire_date DATE
);

CREATE TABLE departments (
    department_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    budget DECIMAL(15,2)
);""",
        "List all employees with their department names, ordered by salary descending"
    ],
    [
        """CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10,2),
    stock INTEGER
);

CREATE TABLE sales (
    sale_id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER,
    sale_date DATE
);""",
        "What is the total revenue for each product category?"
    ],
    [
        """CREATE TABLE students (
    student_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    major VARCHAR(50),
    gpa DECIMAL(3,2)
);

CREATE TABLE courses (
    course_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    credits INTEGER
);

CREATE TABLE enrollments (
    enrollment_id SERIAL PRIMARY KEY,
    student_id INTEGER REFERENCES students(student_id),
    course_id INTEGER REFERENCES courses(course_id),
    grade VARCHAR(2)
);""",
        "Find students who are enrolled in more than 3 courses"
    ],
    [
        """CREATE TABLE authors (
    author_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    country VARCHAR(50)
);

CREATE TABLE books (
    book_id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    author_id INTEGER REFERENCES authors(author_id),
    published_year INTEGER,
    genre VARCHAR(50)
);""",
        "List all authors who have written more than 2 books, along with their book count"
    ]
]


# Create Gradio interface
with gr.Blocks(
    title="Text-to-SQL Generator",
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("""
    # 🔍 Text-to-SQL Generator
    
    Convert natural language questions into PostgreSQL queries using a fine-tuned Mistral-7B model.
    
    **Model:** [rajeshmanikka/mistral-7b-text-to-sql](https://huggingface.co/rajeshmanikka/mistral-7b-text-to-sql)
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            schema_input = gr.Textbox(
                label="Database Schema",
                placeholder="CREATE TABLE customers (\n    customer_id SERIAL PRIMARY KEY,\n    name VARCHAR(100),\n    ...\n);",
                lines=10,
                max_lines=20
            )
            
            question_input = gr.Textbox(
                label="Question",
                placeholder="What are the top 5 customers by total orders?",
                lines=2
            )
            
            generate_btn = gr.Button("Generate SQL", variant="primary")
        
        with gr.Column(scale=1):
            sql_output = gr.Code(
                label="Generated SQL",
                language="sql",
                lines=10
            )
    
    gr.Markdown("""
    ---
    ### 📝 Examples
    Click on any example below to try it:
    """)
    
    gr.Examples(
        examples=EXAMPLES,
        inputs=[schema_input, question_input],
        outputs=sql_output,
        fn=generate_sql,
        cache_examples=False
    )
    
    gr.Markdown("""
    ---
    ### ⚠️ Limitations
    
    - **PostgreSQL Only**: This model generates PostgreSQL syntax (not MySQL or SQLite)
    - **Schema Required**: Always provide the complete database schema
    - **Complex Queries**: Very complex nested subqueries may not always be accurate
    - **Inference Time**: First query may take longer as the model loads (~2-3 minutes on CPU)
    
    ---
    
    **Built with** 🤗 Transformers, PEFT, and Gradio | 
    **Trained on** Spider Dataset | 
    **Base Model** Mistral-7B
    """)
    
    # Connect button to function
    generate_btn.click(
        fn=generate_sql,
        inputs=[schema_input, question_input],
        outputs=sql_output
    )


# Launch configuration
if __name__ == "__main__":
    # Pre-load model on startup (optional, can be slow)
    # load_model()
    
    demo.queue()  # Enable queuing for better handling
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
