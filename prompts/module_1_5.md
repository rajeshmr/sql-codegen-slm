# MODULE 1.5: Create Data Splits and Generate Demo Schemas

**Objective:** Finalize train/validation splits, create a separate test set, and generate 5 demo schemas for the frontend

---

## PROMPT FOR AI IDE (Windsurf/Claude):

### Context
You are working on sql-codegen-slm project. We have train_postgres.jsonl (8,234 examples) and dev_postgres.jsonl (989 examples) from Spider. We need to: (1) Split dev into validation and test sets, (2) Create 5 polished demo schemas for the web UI, (3) Generate comprehensive dataset statistics, and (4) Create data loading utilities for training.

### Task: Finalize Data Pipeline

**Location:** `data/split_creator.py`

**Script Requirements:**

Create a Python module that:

1. **Main function: create_final_splits()**
   - Loads `data/processed/train_postgres.jsonl` (keep as training set)
   - Loads `data/processed/dev_postgres.jsonl`
   - Splits dev into:
     - Validation set (50%): `data/processed/val_postgres.jsonl` (~495 examples)
     - Test set (50%): `data/processed/test_postgres.jsonl` (~494 examples)
   - Uses random seed 42 for reproducibility
   - Ensures balanced split (similar distribution of query complexity)
   - Creates split metadata file: `data/processed/split_info.json`
   - Returns split statistics

2. **Helper function: analyze_query_complexity(sql_query)**
   - Determines query complexity level:
     - Level 1 (Simple): Basic SELECT, no JOIN
     - Level 2 (Medium): With JOIN or GROUP BY
     - Level 3 (Complex): With subquery or multiple JOINs
     - Level 4 (Very Complex): Nested subqueries, multiple aggregations
   - Returns complexity level and features detected

3. **Helper function: stratified_split(examples, split_ratio=0.5)**
   - Analyzes complexity distribution in examples
   - Splits while maintaining similar distributions in both sets
   - Uses stratification by complexity level
   - Returns two lists: validation_examples, test_examples

4. **Validation function: verify_splits()**
   - Checks that splits have:
     - No overlap between train/val/test
     - Similar complexity distributions
     - Similar database diversity
   - Prints comparison statistics
   - Returns True if valid, False otherwise

**Location:** `data/demo_schema_generator.py`

**Script Requirements:**

Create a Python module that:

1. **Main function: create_demo_schemas()**
   - Creates 5 demo schemas for different domains:
     - E-commerce (orders, products, customers, reviews)
     - Finance (accounts, transactions, customers, loans)
     - Healthcare (patients, appointments, doctors, prescriptions)
     - SaaS (users, subscriptions, features, usage_logs)
     - Retail (stores, inventory, sales, employees)
   - Each schema should be:
     - 4-6 tables with relationships
     - Include primary keys and foreign keys
     - PostgreSQL syntax
     - Well-commented
     - Realistic column names
   - Saves to `data/demo/` directory:
     - `ecommerce_schema.sql`
     - `finance_schema.sql`
     - `healthcare_schema.sql`
     - `saas_schema.sql`
     - `retail_schema.sql`
   - Creates `data/demo/demo_schemas.json` with metadata and sample questions
   - Returns list of created schemas

2. **Helper function: generate_sample_questions(schema_name, tables)**
   - For each demo schema, generates 5-10 example questions:
     - Simple queries (e.g., "Show all customers")
     - Queries with JOINs (e.g., "Show orders with customer names")
     - Aggregation queries (e.g., "Total revenue by product category")
     - Complex queries (e.g., "Top 5 customers by purchase amount")
   - Returns list of sample questions with expected SQL patterns

3. **Create demo schemas with this structure for each domain:**

   Example for E-commerce:
   ```sql
   -- E-commerce Database Schema
   -- Description: Online retail platform with orders, products, and customer management
   
   CREATE TABLE customers (
       customer_id SERIAL PRIMARY KEY,
       email VARCHAR(255) UNIQUE NOT NULL,
       name VARCHAR(100) NOT NULL,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   
   CREATE TABLE products (
       product_id SERIAL PRIMARY KEY,
       name VARCHAR(200) NOT NULL,
       category VARCHAR(50),
       price NUMERIC(10, 2),
       stock_quantity INTEGER
   );
   
   CREATE TABLE orders (
       order_id SERIAL PRIMARY KEY,
       customer_id INTEGER REFERENCES customers(customer_id),
       order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       total_amount NUMERIC(10, 2),
       status VARCHAR(20)
   );
   
   CREATE TABLE order_items (
       order_item_id SERIAL PRIMARY KEY,
       order_id INTEGER REFERENCES orders(order_id),
       product_id INTEGER REFERENCES products(product_id),
       quantity INTEGER,
       price NUMERIC(10, 2)
   );
   
   CREATE TABLE reviews (
       review_id SERIAL PRIMARY KEY,
       product_id INTEGER REFERENCES products(product_id),
       customer_id INTEGER REFERENCES customers(customer_id),
       rating INTEGER CHECK (rating >= 1 AND rating <= 5),
       comment TEXT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

**Location:** `data/dataset_stats.py`

**Script Requirements:**

Create a Python module that:

1. **Main function: generate_comprehensive_stats()**
   - Analyzes all three splits (train/val/test)
   - Generates comprehensive statistics:
     - Total examples per split
     - Query complexity distribution
     - Database diversity (unique databases used)
     - SQL keyword frequency (SELECT, JOIN, GROUP BY, etc.)
     - Average schema size (tables per database)
     - Query length distribution (tokens/characters)
   - Saves to `data/processed/dataset_statistics.json`
   - Creates visualization data for charts
   - Prints summary report

2. **Helper function: analyze_sql_features(sql_query)**
   - Extracts SQL features:
     - Has SELECT, INSERT, UPDATE, DELETE
     - Has JOIN (INNER, LEFT, RIGHT)
     - Has WHERE clause
     - Has GROUP BY
     - Has ORDER BY
     - Has LIMIT
     - Has subquery
     - Has aggregation functions (COUNT, SUM, AVG, MAX, MIN)
   - Returns feature dictionary

3. **Helper function: calculate_token_stats(examples)**
   - Calculates:
     - Average tokens in schema
     - Average tokens in question
     - Average tokens in SQL query
     - Max/min lengths
   - Returns statistics dictionary

**Location:** `scripts/finalize_data.sh`

Create bash wrapper that:
1. Activates conda environment
2. Runs split creation
3. Generates demo schemas
4. Computes comprehensive statistics
5. Runs validation
6. Prints final summary
7. Make executable

**Create test file:** `tests/data/test_final_pipeline.py`

Create pytest tests that:
1. Test train/val/test have no overlapping examples
2. Test validation and test sets are roughly equal size
3. Test complexity distribution is similar across val/test
4. Test demo schemas are valid PostgreSQL syntax
5. Test demo schemas have proper foreign key relationships
6. Test all 5 demo schema files exist
7. Test dataset statistics JSON is valid
8. Test data loading utilities work correctly

### Expected Outputs:

**split_info.json:**
```json
{
  "train": {
    "file": "data/processed/train_postgres.jsonl",
    "count": 8234,
    "databases": 146,
    "complexity_distribution": {
      "simple": 2847,
      "medium": 3456,
      "complex": 1534,
      "very_complex": 397
    }
  },
  "validation": {
    "file": "data/processed/val_postgres.jsonl",
    "count": 495,
    "databases": 134,
    "complexity_distribution": {
      "simple": 171,
      "medium": 198,
      "complex": 98,
      "very_complex": 28
    }
  },
  "test": {
    "file": "data/processed/test_postgres.jsonl",
    "count": 494,
    "databases": 132,
    "complexity_distribution": {
      "simple": 168,
      "medium": 201,
      "complex": 96,
      "very_complex": 29
    }
  }
}
```

**demo_schemas.json:**
```json
{
  "ecommerce": {
    "schema_file": "data/demo/ecommerce_schema.sql",
    "name": "E-commerce Platform",
    "description": "Online retail with orders, products, customers",
    "tables": ["customers", "products", "orders", "order_items", "reviews"],
    "sample_questions": [
      "Show all customers who placed orders in the last 30 days",
      "What is the average order value?",
      "List top 5 products by revenue",
      "Show customers with more than 10 orders",
      "Find products with average rating above 4.5"
    ]
  },
  "finance": {
    "schema_file": "data/demo/finance_schema.sql",
    "name": "Banking System",
    "description": "Financial accounts and transactions",
    "tables": ["customers", "accounts", "transactions", "loans", "credit_cards"],
    "sample_questions": [
      "Show total balance across all accounts",
      "List customers with loans over $50,000",
      "Show transaction history for last month",
      "Find accounts with negative balance",
      "Calculate total interest paid on loans"
    ]
  }
}
```

### Testing Requirements:

After creation:
1. Running `./scripts/finalize_data.sh` completes without errors
2. `val_postgres.jsonl` and `test_postgres.jsonl` exist with ~495 examples each
3. All 5 demo schema files exist in `data/demo/`
4. `split_info.json` and `dataset_statistics.json` exist
5. Running `pytest tests/data/test_final_pipeline.py -v` passes all tests
6. Demo schemas are valid and load in PostgreSQL without errors

### Update README.md:

Complete the "Data Processing Pipeline" section:
```
Spider Dataset (SQLite)
    ↓
Module 1.1: Download → data/raw/spider/
    ↓
Module 1.2: Parse Schemas → data/processed/schemas/
    ↓
Module 1.3: Format for Mistral → data/processed/*_mistral.jsonl
    ↓
Module 1.4: Convert to PostgreSQL → data/processed/*_postgres.jsonl
    ↓
Module 1.5: Create Splits + Demo Schemas → READY FOR TRAINING ✅

Final Dataset:
- Training: 8,234 examples
- Validation: 495 examples
- Test: 494 examples
- Demo Schemas: 5 domains (ecommerce, finance, healthcare, saas, retail)
```

Add "Demo Schemas" section:
- List 5 demo schemas with descriptions
- Explain they're for frontend demonstration
- Show sample questions for each

### Commit Message:
"feat(data): Finalize data splits and generate demo schemas - Module 1.5"

---

## ✅ MODULE 1.5 COMPLETION CHECKLIST

**After running the AI IDE prompt, verify the following:**

### Files Created:
- [ ] `data/split_creator.py` exists
- [ ] `data/demo_schema_generator.py` exists
- [ ] `data/dataset_stats.py` exists
- [ ] `scripts/finalize_data.sh` exists and is executable
- [ ] `tests/data/test_final_pipeline.py` exists
- [ ] README.md updated with complete pipeline and demo schemas

### Output Files - Splits:
- [ ] `data/processed/train_postgres.jsonl` exists (8,234 examples)
- [ ] `data/processed/val_postgres.jsonl` exists (~495 examples)
- [ ] `data/processed/test_postgres.jsonl` exists (~494 examples)
- [ ] `data/processed/split_info.json` exists
- [ ] Val + Test = original dev.jsonl line count (989)

### Output Files - Demo Schemas:
- [ ] `data/demo/` directory exists
- [ ] `data/demo/ecommerce_schema.sql` exists
- [ ] `data/demo/finance_schema.sql` exists
- [ ] `data/demo/healthcare_schema.sql` exists
- [ ] `data/demo/saas_schema.sql` exists
- [ ] `data/demo/retail_schema.sql` exists
- [ ] `data/demo/demo_schemas.json` exists with metadata

### Output Files - Statistics:
- [ ] `data/processed/dataset_statistics.json` exists
- [ ] Contains stats for train/val/test splits
- [ ] Contains complexity distributions
- [ ] Contains SQL feature analysis

### Split Validation:

Verify no overlap:
```python
import json

def load_queries(filepath):
    queries = set()
    with open(filepath) as f:
        for line in f:
            ex = json.loads(line)
            queries.add(ex['messages'][2]['content'])  # SQL query
    return queries

train = load_queries('data/processed/train_postgres.jsonl')
val = load_queries('data/processed/val_postgres.jsonl')
test = load_queries('data/processed/test_postgres.jsonl')

print(f"Train-Val overlap: {len(train & val)}")  # Should be 0
print(f"Train-Test overlap: {len(train & test)}")  # Should be 0
print(f"Val-Test overlap: {len(val & test)}")  # Should be 0
```

- [ ] No overlap between train and validation
- [ ] No overlap between train and test
- [ ] No overlap between validation and test

### Split Distribution Check:
- [ ] Open `split_info.json`
- [ ] Validation and test have similar complexity distributions
- [ ] Validation and test have similar database diversity
- [ ] Both sets have mix of simple/medium/complex queries (not all simple)

### Demo Schema Validation:

Open each demo schema:
- [ ] E-commerce schema: Has customers, products, orders tables with foreign keys
- [ ] Finance schema: Has accounts, transactions, customers tables
- [ ] Healthcare schema: Has patients, doctors, appointments tables
- [ ] SaaS schema: Has users, subscriptions, features tables
- [ ] Retail schema: Has stores, inventory, sales tables

For each schema:
- [ ] Valid PostgreSQL syntax (no SQLite-isms)
- [ ] Uses SERIAL for auto-increment
- [ ] Has REFERENCES for foreign keys
- [ ] Has descriptive comments
- [ ] 4-6 tables per schema
- [ ] Realistic column names and types

### Demo Schemas JSON Check:
- [ ] `demo_schemas.json` has entries for all 5 schemas
- [ ] Each entry has: name, description, tables list, sample_questions
- [ ] Sample questions are realistic and diverse
- [ ] Questions range from simple to complex

### Statistics Validation:
- [ ] Open `dataset_statistics.json`
- [ ] Shows total examples per split
- [ ] Shows query complexity distribution
- [ ] Shows SQL keyword frequencies (SELECT, JOIN, GROUP BY)
- [ ] Shows average query length (tokens/characters)
- [ ] Shows database diversity metrics

### Functional Tests:
- [ ] Run `pytest tests/data/test_final_pipeline.py -v`
- [ ] All 8+ tests pass
- [ ] No overlap test passes
- [ ] Split size tests pass
- [ ] Demo schema validation passes

### Manual Demo Schema Test:

Try loading one demo schema in PostgreSQL (if you have psql):
```bash
psql -U postgres -d test_db < data/demo/ecommerce_schema.sql
```

Or just verify syntax:
- [ ] Can read schema files without errors
- [ ] CREATE TABLE statements look correct
- [ ] Foreign keys reference existing tables
- [ ] No syntax errors visible

### Understanding Check (Learning):
- [ ] You understand why we split dev into val and test (validation during training, final evaluation)
- [ ] You know why stratified splitting matters (maintains distribution)
- [ ] You understand why demo schemas are separate from training data (for frontend demos)
- [ ] You know what makes a good demo schema (realistic, 4-6 tables, relationships)
- [ ] You understand the full data pipeline end-to-end (download → parse → format → convert → split)

### Data Pipeline Complete Check:

Verify the full pipeline:
```
✅ Module 1.1: Spider dataset downloaded (200 databases)
✅ Module 1.2: Schemas parsed (166 schemas)
✅ Module 1.3: Formatted for Mistral (9,223 examples)
✅ Module 1.4: Converted to PostgreSQL (9,223 examples)
✅ Module 1.5: Split into train/val/test + demo schemas
```

Final counts:
- [ ] Training: 8,234 examples
- [ ] Validation: ~495 examples
- [ ] Test: ~494 examples
- [ ] Total: 9,223 examples
- [ ] Demo schemas: 5 domains

### Git Verification:
- [ ] Split JSONL files NOT tracked (.gitignore)
- [ ] Demo schema SQL files ARE tracked (small, useful for repo)
- [ ] demo_schemas.json IS tracked
- [ ] split_info.json IS tracked
- [ ] dataset_statistics.json IS tracked
- [ ] Code files (Python, tests, scripts) are staged
- [ ] Commit message follows convention

### What You Should Have Learned:
1. **Data Splitting**: Importance of train/validation/test separation
2. **Stratification**: Maintaining distribution across splits
3. **Reproducibility**: Using random seeds for consistent splits
4. **Demo Data**: Creating realistic schemas for product demos
5. **Data Pipeline**: Building robust, testable data preparation pipelines
6. **Validation**: Verifying no data leakage between splits
7. **Documentation**: Importance of comprehensive dataset statistics

---

**TROUBLESHOOTING (if checks fail):**

❌ **Overlap detected between splits:**
- Bug in splitting logic
- Re-run with fresh random seed
- Verify set operations are correct

❌ **Unbalanced splits:**
- Check stratification logic
- Verify complexity analysis is working
- May need to adjust split ratio

❌ **Demo schemas have syntax errors:**
- Test each schema in PostgreSQL
- Check foreign key references
- Verify column types are valid PostgreSQL types

❌ **Tests fail:**
- Check file paths in tests
- Verify JSONL format is correct
- Ensure test data fixtures exist

---

**SAMPLE OUTPUT YOU SHOULD SEE:**

When running `./scripts/finalize_data.sh`:
```
Finalizing data splits and creating demo schemas...

Creating validation and test splits...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loaded dev set: 989 examples
Analyzing complexity distribution...
  Simple: 339 (34.3%)
  Medium: 399 (40.3%)
  Complex: 194 (19.6%)
  Very Complex: 57 (5.8%)

Performing stratified split (50/50)...
✅ Validation set: 495 examples
✅ Test set: 494 examples

Verifying no overlap... ✅ Passed
Verifying distributions match... ✅ Passed (within 2%)

Generating demo schemas...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ E-commerce schema (5 tables, 10 sample questions)
✅ Finance schema (5 tables, 10 sample questions)
✅ Healthcare schema (6 tables, 10 sample questions)
✅ SaaS schema (5 tables, 10 sample questions)
✅ Retail schema (5 tables, 10 sample questions)

Computing dataset statistics...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training Set:
  Examples: 8,234
  Databases: 146
  Avg query length: 47 tokens
  Complex queries: 23.5%

Validation Set:
  Examples: 495
  Databases: 134
  Avg query length: 46 tokens
  Complex queries: 25.5%

Test Set:
  Examples: 494
  Databases: 132
  Avg query length: 48 tokens
  Complex queries: 24.3%

✅ DATA PIPELINE COMPLETE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total training examples: 8,234
Total validation examples: 495
Total test examples: 494
Demo schemas created: 5

Ready for Module 2: Model Training 🚀
```