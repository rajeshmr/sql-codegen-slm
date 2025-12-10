# MODULE 1.4: Convert SQLite Queries to PostgreSQL Dialect

**Objective:** Transform SQLite SQL syntax to PostgreSQL-compatible syntax for both queries and schemas

---

## PROMPT FOR AI IDE (Windsurf/Claude):

### Context
You are working on sql-codegen-slm project. Spider dataset uses SQLite databases and SQLite SQL syntax. Since we're targeting PostgreSQL as specified in the system message, we need to convert SQLite-specific syntax to PostgreSQL. This includes data types, functions, operators, and keywords that differ between the two dialects.

### Task: Create SQLite to PostgreSQL Converter

**Location:** `data/postgres_converter.py`

**Script Requirements:**

Create a Python module that:

1. **Main function: convert_to_postgres()**
   - Loads `data/processed/train_mistral.jsonl`
   - Loads `data/processed/dev_mistral.jsonl`
   - For each example:
     - Extracts the assistant's SQL query
     - Extracts the schema from user message
     - Converts both SQL query and schema to PostgreSQL syntax
     - Updates the example with converted SQL
     - Updates schema in user message with converted schema
   - Saves to:
     - `data/processed/train_postgres.jsonl`
     - `data/processed/dev_postgres.jsonl`
   - Shows progress bar
   - Returns conversion statistics (queries converted, patterns found)

2. **Helper function: convert_sql_query(sql_query)**
   - Takes SQLite SQL query string
   - Applies conversion rules (see below)
   - Returns PostgreSQL-compatible SQL string
   - Handles edge cases gracefully

3. **Helper function: convert_schema(schema_text)**
   - Takes SQLite schema (CREATE TABLE statements)
   - Converts SQLite-specific syntax to PostgreSQL
   - Returns PostgreSQL-compatible schema string

4. **Conversion rules to implement:**

   **Data Types:**
   - `INTEGER AUTOINCREMENT` → `SERIAL`
   - `INTEGER PRIMARY KEY AUTOINCREMENT` → `SERIAL PRIMARY KEY`
   - `TEXT` → `TEXT` (same in both)
   - `REAL` → `NUMERIC` or `DOUBLE PRECISION`
   - `BLOB` → `BYTEA`

   **String Functions:**
   - `||` (concatenation) → `CONCAT()` or keep `||` (PostgreSQL supports both, but CONCAT is more explicit)
   - `SUBSTR(str, start, length)` → `SUBSTRING(str FROM start FOR length)`
   - `LENGTH(str)` → `LENGTH(str)` (same)
   - `UPPER(str)` → `UPPER(str)` (same)
   - `LOWER(str)` → `LOWER(str)` (same)

   **Date/Time Functions:**
   - `DATETIME('now')` → `CURRENT_TIMESTAMP`
   - `DATE('now')` → `CURRENT_DATE`
   - `TIME('now')` → `CURRENT_TIME`
   - `strftime(format, date)` → `TO_CHAR(date, format)` (format strings differ, handle common ones)

   **Boolean Values:**
   - `1` (in boolean context) → `TRUE`
   - `0` (in boolean context) → `FALSE`
   - Note: Only convert when clearly boolean (e.g., `is_active = 1`)

   **PRAGMA Statements:**
   - `PRAGMA foreign_keys = ON;` → Remove (PostgreSQL enforces by default)
   - Other PRAGMA statements → Remove or convert to PostgreSQL equivalents

   **Quote Styles:**
   - SQLite uses `"column"` for identifiers
   - PostgreSQL prefers unquoted or `"column"` (both work)
   - Keep as-is unless causing issues

5. **Helper function: detect_conversion_patterns(sql_query)**
   - Identifies which SQLite patterns are present in query
   - Returns dictionary: {"concatenation": bool, "date_functions": bool, ...}
   - Used for statistics and validation

6. **Validation function: validate_postgres_syntax(sql_query)**
   - Basic syntax validation for PostgreSQL
   - Checks for common SQLite-only syntax that wasn't converted
   - Returns tuple: (is_valid, issues_found)
   - Don't need full SQL parser, just pattern matching

7. **Statistics function: analyze_conversions()**
   - Counts conversion patterns applied:
     - AUTOINCREMENT → SERIAL conversions
     - || operator conversions
     - Date function conversions
     - PRAGMA removals
   - Saves to `data/processed/postgres_conversion_stats.json`
   - Prints summary

**Location:** `scripts/convert_to_postgres.sh`

Create bash wrapper that:
1. Activates conda environment
2. Runs `python -m data.postgres_converter`
3. Runs validation on sample of converted queries
4. Prints statistics
5. Shows before/after examples
6. Make executable

**Create test file:** `tests/data/test_postgres_converter.py`

Create pytest tests that:
1. Test convert_sql_query() with SQLite concatenation: `'a' || 'b'` → verify conversion
2. Test AUTOINCREMENT conversion in schemas
3. Test date function conversion: `DATETIME('now')` → `CURRENT_TIMESTAMP`
4. Test PRAGMA removal from schemas
5. Test that valid PostgreSQL queries pass validation
6. Test that unconverted SQLite syntax is detected
7. Test end-to-end: load example, convert, verify structure intact
8. Test edge case: query with multiple conversion patterns

**Create conversion reference:** `docs/sqlite_to_postgres.md`

Document all conversion rules with examples:
```markdown
# SQLite to PostgreSQL Conversion Reference

## Data Types
- `INTEGER AUTOINCREMENT` → `SERIAL`
  - Before: `CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT)`
  - After: `CREATE TABLE t (id SERIAL PRIMARY KEY)`

## String Operations
- Concatenation: `||` → `CONCAT()` (optional, both work)
  - Before: `SELECT first_name || ' ' || last_name`
  - After: `SELECT CONCAT(first_name, ' ', last_name)` or keep as-is

... [full reference]
```

### Example Conversion:

**Before (SQLite):**
```sql
-- Schema
PRAGMA foreign_keys = ON;
CREATE TABLE "author" (
"aid" INTEGER PRIMARY KEY AUTOINCREMENT,
"name" TEXT,
"created_at" DATETIME DEFAULT DATETIME('now')
);

-- Query
SELECT "name" || ' (' || "aid" || ')' FROM author WHERE created_at > DATETIME('now', '-7 days');
```

**After (PostgreSQL):**
```sql
-- Schema
CREATE TABLE "author" (
"aid" SERIAL PRIMARY KEY,
"name" TEXT,
"created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Query
SELECT CONCAT("name", ' (', "aid", ')') FROM author WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days';
```

### Testing Requirements:

After creation:
1. Running `./scripts/convert_to_postgres.sh` converts all examples
2. `train_postgres.jsonl` and `dev_postgres.jsonl` exist with same line count as originals
3. Converted queries have PostgreSQL syntax
4. Schemas have PRAGMA statements removed
5. Running `pytest tests/data/test_postgres_converter.py -v` passes all tests
6. Conversion statistics show patterns detected and converted

### Update README.md:

Update "Data Processing Pipeline" section:
```
Spider Dataset (SQLite)
    ↓
Module 1.1: Download → data/raw/spider/
    ↓
Module 1.2: Parse Schemas → data/processed/schemas/
    ↓
Module 1.3: Format for Mistral → data/processed/*_mistral.jsonl
    ↓
Module 1.4: Convert to PostgreSQL → data/processed/*_postgres.jsonl ← YOU ARE HERE
    ↓
Module 1.5: Create splits (next)
```

### Commit Message:
"feat(data): Add SQLite to PostgreSQL converter for queries and schemas - Module 1.4"

---

## ✅ MODULE 1.4 COMPLETION CHECKLIST

**After running the AI IDE prompt, verify the following:**

### Files Created:
- [ ] `data/postgres_converter.py` exists
- [ ] `scripts/convert_to_postgres.sh` exists and is executable
- [ ] `tests/data/test_postgres_converter.py` exists
- [ ] `docs/sqlite_to_postgres.md` exists with conversion reference
- [ ] README.md updated with pipeline diagram

### Output Files:
- [ ] `data/processed/train_postgres.jsonl` exists
- [ ] `data/processed/dev_postgres.jsonl` exists
- [ ] `data/processed/postgres_conversion_stats.json` exists

### File Validation:
- [ ] `train_postgres.jsonl` has same number of lines as `train_mistral.jsonl`
- [ ] `dev_postgres.jsonl` has same number of lines as `dev_mistral.jsonl`
- [ ] File sizes similar to originals (±10%)

### Conversion Verification:

Pick 3 random examples and verify:
- [ ] Open `train_postgres.jsonl`, pick random line
- [ ] Check assistant's SQL query - should NOT have `DATETIME('now')`
- [ ] Should have `CURRENT_TIMESTAMP` instead
- [ ] Check schema in user message - should NOT have `PRAGMA foreign_keys`
- [ ] Check schema - should NOT have `AUTOINCREMENT`
- [ ] Should have `SERIAL` for auto-increment columns
- [ ] SQL query syntax looks reasonable (no obvious errors)

### Pattern Detection:

Check `postgres_conversion_stats.json`:
- [ ] Shows count of AUTOINCREMENT conversions (should be 100+)
- [ ] Shows count of PRAGMA removals (should be 150+)
- [ ] Shows count of date function conversions (should be 50+)
- [ ] Shows count of concatenation patterns found (may be 0 if kept as-is)
- [ ] Total conversions > 0 (proves converter is working)

### Before/After Comparison:

Manually compare one example:
```bash
# Get same example from both files (line 100)
sed -n '100p' data/processed/train_mistral.jsonl > /tmp/before.json
sed -n '100p' data/processed/train_postgres.jsonl > /tmp/after.json

# Pretty print and compare
python -m json.tool /tmp/before.json > /tmp/before_pretty.json
python -m json.tool /tmp/after.json > /tmp/after_pretty.json
diff /tmp/before_pretty.json /tmp/after_pretty.json
```

Verify:
- [ ] System and user question unchanged
- [ ] Only SQL query and schema modified
- [ ] Conversions applied correctly

### Functional Tests:
- [ ] Run `pytest tests/data/test_postgres_converter.py -v`
- [ ] All 8+ tests pass
- [ ] Test for concatenation conversion works
- [ ] Test for AUTOINCREMENT conversion works
- [ ] Test for date function conversion works
- [ ] Test for PRAGMA removal works

### Schema Conversion Check:

Extract and inspect one converted schema:
```python
import json
with open('data/processed/train_postgres.jsonl', 'r') as f:
    example = json.loads(f.readline())
    schema = example['messages'][1]['content'].split('Schema:')[1].split('Question:')[0]
    print(schema)
```

Verify:
- [ ] No `PRAGMA` statements
- [ ] `SERIAL` instead of `AUTOINCREMENT`
- [ ] `CURRENT_TIMESTAMP` instead of `DATETIME('now')`
- [ ] Otherwise schema structure intact

### Understanding Check (Learning):
- [ ] You understand why dialect conversion is needed (SQLite ≠ PostgreSQL)
- [ ] You know key differences: AUTOINCREMENT vs SERIAL
- [ ] You understand date function differences between dialects
- [ ] You know why PRAGMA statements are SQLite-specific
- [ ] You recognize that some syntax is compatible (SELECT, JOIN, WHERE mostly same)

### Edge Cases Handled:
- [ ] Queries with no conversions needed (pure standard SQL) - unchanged
- [ ] Queries with multiple patterns - all converted
- [ ] Schemas without AUTOINCREMENT - unchanged except PRAGMA removal
- [ ] Empty or malformed queries - handled gracefully (skipped or error logged)

### Documentation Check:
- [ ] `docs/sqlite_to_postgres.md` exists
- [ ] Has clear before/after examples
- [ ] Documents all conversion rules
- [ ] Explains when conversions are optional vs required

### Git Verification:
- [ ] `data/processed/*_postgres.jsonl` files are NOT tracked (.gitignore)
- [ ] Code files (postgres_converter.py, tests, scripts) are staged
- [ ] Documentation (sqlite_to_postgres.md) is staged
- [ ] Conversion stats JSON is tracked (small metadata file)
- [ ] Commit message follows convention

### What You Should Have Learned:
1. **SQL Dialects**: Different databases have different syntax (not all SQL is the same)
2. **Auto-increment**: SQLite uses AUTOINCREMENT, PostgreSQL uses SERIAL
3. **Date Functions**: Each database has its own date/time function names
4. **PRAGMA**: SQLite-specific commands for configuration
5. **String Concatenation**: Multiple ways to do same thing (|| vs CONCAT)
6. **Pattern Matching**: Using regex to detect and replace SQL patterns
7. **Lossless Conversion**: Keep semantic meaning while changing syntax

---

**TROUBLESHOOTING (if checks fail):**

❌ **Line counts don't match:**
- Check if converter is skipping examples with errors
- Verify JSONL reading/writing logic (each line should be complete JSON)
- Some examples might fail conversion - that's okay if <1%

❌ **Conversions not applied:**
- Check regex patterns in conversion functions
- Verify patterns are case-insensitive where needed
- Test on isolated examples to debug

❌ **Queries look broken after conversion:**
- Review conversion rules - might be too aggressive
- Some SQLite syntax might need manual handling
- Check if standard SQL was incorrectly modified

❌ **Tests fail:**
- Verify test input examples use SQLite syntax
- Check if conversion functions return expected output format
- Ensure edge cases are handled (None, empty strings)

---

**SAMPLE OUTPUT YOU SHOULD SEE:**

When running `./scripts/convert_to_postgres.sh`:
```
Converting SQLite syntax to PostgreSQL...

Processing train_mistral.jsonl: 100%|████| 8234/8234 [00:08<00:00, 1029.25it/s]
Processing dev_mistral.jsonl: 100%|████████| 989/989 [00:01<00:00, 987.43it/s]

✅ Conversion Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training examples converted: 8,234
Validation examples converted: 989

Conversion Statistics:
- AUTOINCREMENT → SERIAL: 187 conversions
- PRAGMA statements removed: 166 schemas
- Date functions converted: 73 queries
  - DATETIME('now') → CURRENT_TIMESTAMP: 45
  - DATE('now') → CURRENT_DATE: 28
- String concatenation (||): 234 found (kept as-is, PostgreSQL compatible)

Output files:
- data/processed/train_postgres.jsonl (8,234 examples)
- data/processed/dev_postgres.jsonl (989 examples)

Validation: Checking 100 random examples...
✅ All checked examples have valid PostgreSQL syntax

Sample Conversion:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before (SQLite):
  Schema: CREATE TABLE author (aid INTEGER PRIMARY KEY AUTOINCREMENT, ...);
  Query: SELECT * FROM author WHERE created_at > DATETIME('now', '-7 days');

After (PostgreSQL):
  Schema: CREATE TABLE author (aid SERIAL PRIMARY KEY, ...);
  Query: SELECT * FROM author WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days';
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```