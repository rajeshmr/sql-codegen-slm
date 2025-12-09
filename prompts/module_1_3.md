# MODULE 1.3: Convert Spider to Mistral Instruction Format

**Objective:** Transform Spider JSON data into Mistral instruction tuning format (JSONL) with schema context

---

## PROMPT FOR AI IDE (Windsurf/Claude):

### Context
You are working on sql-codegen-slm project. Spider dataset stores examples in JSON format with separate fields for question, SQL query, and database ID. Mistral fine-tuning requires instruction format with system/user/assistant message roles. We need to combine the question with its corresponding schema and format as conversational turns.

### Task: Create Spider to Mistral Converter

**Location:** `data/format_converter.py`

**Script Requirements:**

Create a Python module that:

1. **Main function: convert_spider_to_mistral()**
   - Loads `data/raw/spider/train_spider.json` (8,659 examples)
   - Loads `data/raw/spider/dev.json` (1,034 examples)
   - Loads `data/processed/schemas/schema_index.json` (schema metadata)
   - For each Spider example:
     - Extract: question, SQL query, database ID
     - Load corresponding schema from `data/processed/schemas/{db_id}_schema.sql`
     - Format as Mistral instruction format (see structure below)
     - Handle missing schemas gracefully (skip those examples with warning)
   - Saves to:
     - `data/processed/train_mistral.jsonl` (training examples)
     - `data/processed/dev_mistral.jsonl` (validation examples)
   - Shows progress bar
   - Returns conversion statistics

2. **Helper function: format_mistral_example(question, sql_query, schema_content, db_id)**
   - Takes question, SQL, schema text, database name
   - Creates dictionary in Mistral format:
     ```python
     {
       "messages": [
         {
           "role": "system",
           "content": "You are an expert PostgreSQL query generator. Given a database schema and a natural language question, generate a valid SQL query."
         },
         {
           "role": "user",
           "content": f"Database: {db_id}\n\nSchema:\n{schema_content}\n\nQuestion: {question}\n\nGenerate the SQL query:"
         },
         {
           "role": "assistant",
           "content": sql_query
         }
       ]
     }
     ```
   - Returns formatted dictionary

3. **Helper function: load_schema_content(db_id, schema_index)**
   - Takes database ID and schema index dictionary
   - Reads schema file from disk
   - Returns schema content as string
   - Returns None if schema not found

4. **Helper function: clean_sql_query(sql_query)**
   - Removes extra whitespace
   - Ensures query ends with semicolon
   - Normalizes to single line or keeps multi-line (your choice for readability)
   - Returns cleaned SQL string

5. **Validation function: validate_conversion()**
   - Loads one JSONL file
   - Checks first 10 examples have correct structure
   - Verifies all required fields present ("messages", "role", "content")
   - Prints sample formatted example
   - Returns True if valid, False otherwise

6. **Statistics function: analyze_dataset()**
   - Counts examples by query complexity:
     - Simple SELECT (no JOIN): count
     - With JOIN: count
     - With GROUP BY: count
     - With subquery: count
     - With ORDER BY/LIMIT: count
   - Prints distribution statistics
   - Saves to `data/processed/dataset_statistics.json`

**Location:** `scripts/convert_to_mistral.sh`

Create bash wrapper that:
1. Activates conda environment
2. Runs `python -m data.format_converter`
3. Runs validation
4. Prints statistics
5. Shows sample formatted example
6. Make executable

**Create test file:** `tests/data/test_format_converter.py`

Create pytest tests that:
1. Test format_mistral_example() produces correct structure
2. Test clean_sql_query() removes extra whitespace
3. Test load_schema_content() reads academic schema correctly
4. Test validation catches malformed examples
5. Test JSONL files are valid JSON per line
6. Test that number of examples in JSONL matches input JSON (approximately, accounting for missing schemas)
7. Test schema content is properly embedded in user message

### Expected Mistral Format Example:

Input from Spider:
```json
{
  "db_id": "academic",
  "question": "How many authors are there?",
  "query": "SELECT count(*) FROM author"
}
```

Output in train_mistral.jsonl (single line):
```json
{"messages": [{"role": "system", "content": "You are an expert PostgreSQL query generator. Given a database schema and a natural language question, generate a valid SQL query."}, {"role": "user", "content": "Database: academic\n\nSchema:\nPRAGMA foreign_keys = ON;\nCREATE TABLE \"author\" (\n\"aid\" int,\n\"homepage\" text,\n\"name\" text,\n\"oid\" int,\nprimary key(\"aid\")\n);\n...[rest of schema]...\n\nQuestion: How many authors are there?\n\nGenerate the SQL query:"}, {"role": "assistant", "content": "SELECT count(*) FROM author;"}]}
```

Note: Each line is a complete JSON object, files are newline-delimited JSON (JSONL format).

### Testing Requirements:

After creation:
1. Running `./scripts/convert_to_mistral.sh` converts all examples
2. `train_mistral.jsonl` has ~8,000+ lines (some examples skipped if schema missing)
3. `dev_mistral.jsonl` has ~1,000+ lines
4. Each line is valid JSON
5. Running `pytest tests/data/test_format_converter.py -v` passes all tests
6. Sample output shows properly formatted instruction with schema included

### Update README.md:

Add section "Data Processing Pipeline" showing:
```
Spider Dataset
    ↓
Module 1.1: Download → data/raw/spider/
    ↓
Module 1.2: Parse Schemas → data/processed/schemas/
    ↓
Module 1.3: Format for Mistral → data/processed/*.jsonl ← YOU ARE HERE
    ↓
Module 1.4: Convert to PostgreSQL (next)
```

### Commit Message:
"feat(data): Convert Spider examples to Mistral instruction format - Module 1.3"

---

## ✅ MODULE 1.3 COMPLETION CHECKLIST

**After running the AI IDE prompt, verify the following:**

### Files Created:
- [ ] `data/format_converter.py` exists
- [ ] `scripts/convert_to_mistral.sh` exists and is executable
- [ ] `tests/data/test_format_converter.py` exists
- [ ] README.md updated with data pipeline diagram

### Output Files:
- [ ] `data/processed/train_mistral.jsonl` exists
- [ ] `data/processed/dev_mistral.jsonl` exists
- [ ] `data/processed/dataset_statistics.json` exists

### File Validation:
- [ ] `train_mistral.jsonl` has 7,500-8,500 lines (some examples may be skipped)
- [ ] `dev_mistral.jsonl` has 900-1,034 lines
- [ ] File size reasonable: train_mistral.jsonl should be 50-150MB
- [ ] Each line is valid JSON (no syntax errors)

### Format Verification:
- [ ] Open `train_mistral.jsonl` in text editor
- [ ] Pick a random line, verify it has "messages" array
- [ ] Verify "messages" has 3 elements (system, user, assistant)
- [ ] System message mentions "PostgreSQL query generator"
- [ ] User message includes: Database name, Schema (full CREATE TABLE statements), Question
- [ ] Assistant message contains SQL query with semicolon

### Content Quality Checks:
- [ ] Pick 5 random examples, verify schema matches database
- [ ] Check that schema includes all tables for that database
- [ ] Verify SQL query is syntactically reasonable (no obvious errors)
- [ ] Check that question matches SQL query intent
- [ ] Confirm schema is complete (has CREATE TABLE statements, not truncated)

### Statistics Validation:
- [ ] Open `dataset_statistics.json`
- [ ] Should show breakdown:
  - Total examples processed
  - Examples skipped (missing schema)
  - Simple SELECT queries: count
  - Queries with JOINs: count
  - Queries with GROUP BY: count
  - Queries with subqueries: count
- [ ] Verify that complex queries (JOINs, subqueries) exist (not just simple SELECTs)

### Functional Tests:
- [ ] Run `pytest tests/data/test_format_converter.py -v`
- [ ] All 7+ tests pass
- [ ] Test validates JSON structure
- [ ] Test validates schema content is embedded
- [ ] Test validates SQL cleaning works

### Sample Output Check:

Run this manually to inspect one example:
```python
import json
with open('data/processed/train_mistral.jsonl', 'r') as f:
    example = json.loads(f.readline())
    print(json.dumps(example, indent=2))
```

Should see:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert PostgreSQL query generator..."
    },
    {
      "role": "user",
      "content": "Database: academic\n\nSchema:\n[CREATE TABLE statements]...\n\nQuestion: How many authors are there?"
    },
    {
      "role": "assistant",
      "content": "SELECT count(*) FROM author;"
    }
  ]
}
```

### Understanding Check (Learning):
- [ ] You understand why we use instruction format (conversational turns for LLM training)
- [ ] You know what JSONL is (JSON Lines: one JSON object per line)
- [ ] You understand why schema is in user message (provides context for SQL generation)
- [ ] You know the three roles: system (instructions), user (input), assistant (output)
- [ ] You understand why some examples are skipped (missing schemas)

### Data Distribution Insights:
Answer these by looking at statistics:
- [ ] What percentage of queries are simple SELECT? (should be <40%)
- [ ] What percentage have JOINs? (should be >30%)
- [ ] What percentage have GROUP BY? (should be >20%)
- [ ] Are there subqueries? (should be >10%)
- [ ] This distribution shows Spider is complex, not just trivial queries

### Git Verification:
- [ ] `data/processed/*.jsonl` files are NOT tracked (.gitignore should exclude)
- [ ] Code files (format_converter.py, tests, scripts) are staged
- [ ] dataset_statistics.json IS tracked (small metadata file)
- [ ] Commit message follows convention

### What You Should Have Learned:
1. **Instruction Tuning Format**: How LLMs are trained on conversational format (system/user/assistant)
2. **Context Window Management**: Schema goes in user message to provide necessary context
3. **JSONL Format**: Efficient format for large training datasets (streaming, parallel processing)
4. **Data Cleaning**: Importance of normalizing SQL (whitespace, semicolons)
5. **Dataset Statistics**: Understanding query complexity distribution (simple vs complex)

---

**TROUBLESHOOTING (if checks fail):**

❌ **Fewer examples than expected:**
- Check how many schemas are missing (acceptable: 5-10%)
- Verify schema_index.json has correct file paths
- Some databases in Spider might not have schema files

❌ **JSON parsing errors in JSONL:**
- Ensure each line is a complete JSON object (no line breaks within JSON)
- Check for special characters in schema/questions that need escaping
- Verify quotes are properly escaped in SQL queries

❌ **Schema content looks truncated:**
- Check file reading logic (read full file, not just first line)
- Verify schema files aren't corrupted
- Check memory limits if files are large

❌ **SQL queries don't match questions:**
- This is a Spider dataset issue, not your code
- Spot-check a few examples, should make sense
- If many don't match, check if you're reading correct fields from Spider JSON

---

**SAMPLE OUTPUT YOU SHOULD SEE:**

When running `./scripts/convert_to_mistral.sh`:
```
Converting Spider examples to Mistral format...

Processing train_spider.json: 100%|████| 8659/8659 [00:12<00:00, 721.58it/s]
Processing dev.json: 100%|████████████| 1034/1034 [00:01<00:00, 689.32it/s]

✅ Conversion Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training examples processed: 8,659
Training examples written: 8,234 (95.1%)
Skipped (missing schema): 425 (4.9%)

Validation examples processed: 1,034
Validation examples written: 989 (95.6%)
Skipped (missing schema): 45 (4.4%)

Output files:
- data/processed/train_mistral.jsonl (8,234 examples, 127 MB)
- data/processed/dev_mistral.jsonl (989 examples, 15 MB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset Statistics:
Simple SELECT (no JOIN): 2,847 (34.6%)
With JOIN: 2,965 (36.0%)
With GROUP BY: 1,876 (22.8%)
With subquery: 546 (6.6%)

Validating format...
✅ All examples have correct structure
✅ Schema content properly embedded

Sample example:
Database: concert_singer
Question: How many singers do we have?
Schema: [14 CREATE TABLE statements]
SQL: SELECT count(*) FROM singer;
```