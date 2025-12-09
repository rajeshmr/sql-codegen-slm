# MODULE 1.2 (REVISED): Parse Existing Schema Files

**Objective:** Read and organize the pre-existing schema.sql files from Spider dataset

---

## PROMPT FOR AI IDE (Windsurf/Claude):

### Context
You are working on sql-codegen-slm project. The Spider dataset provides pre-written schema.sql files in each database directory (e.g., `data/raw/spider/database/academic/schema.sql`). These files contain CREATE TABLE statements with foreign key constraints and are already properly formatted. We just need to read, validate, and organize them for training.

### Task: Parse and Index Existing Schema Files

**Location:** `data/schema_parser.py`

**Script Requirements:**

Create a Python module that:

1. **Main function: parse_all_schemas()**
   - Scans `data/raw/spider/database/` for all subdirectories
   - For each subdirectory, finds and reads `schema.sql` file
   - Copies schema file to `data/processed/schemas/{db_name}_schema.sql` (for organization)
   - Parses each schema to extract metadata:
     - List of table names (from CREATE TABLE statements)
     - Number of tables
     - Table with columns mapping
     - Foreign key relationships
   - Creates master index `data/processed/schemas/schema_index.json` with:
     ```json
     {
       "academic": {
         "schema_file": "data/processed/schemas/academic_schema.sql",
         "source_file": "data/raw/spider/database/academic/schema.sql",
         "num_tables": 14,
         "tables": ["author", "conference", "domain", ...],
         "has_foreign_keys": true,
         "parsed_timestamp": "2024-12-09T10:30:00"
       }
     }
     ```
   - Shows progress bar using tqdm
   - Returns statistics: total databases found, successfully parsed, missing schema files, parse errors

2. **Helper function: parse_schema_file(schema_path)**
   - Reads schema.sql file content
   - Extracts table names using regex: `CREATE TABLE ["']?(\w+)["']?`
   - Counts number of CREATE TABLE statements
   - Detects foreign key constraints: looks for "foreign key" in file
   - Returns dictionary with parsed metadata

3. **Helper function: validate_schema_content(schema_content)**
   - Checks if schema contains valid SQL syntax (basic validation)
   - Verifies it has at least one CREATE TABLE statement
   - Checks for PRAGMA statements (should start with PRAGMA foreign_keys)
   - Returns boolean: valid or not, plus any warnings

4. **Validation function: cross_reference_tables_json()**
   - Reads `data/raw/spider/tables.json`
   - Compares database names and table names with parsed schemas
   - Reports any mismatches (database in tables.json but no schema.sql found)
   - Prints validation summary

**Location:** `scripts/parse_schemas.sh`

Create bash wrapper that:
1. Activates conda environment
2. Creates `data/processed/schemas/` directory
3. Runs `python -m data.schema_parser`
4. Prints summary statistics
5. Runs validation
6. Make executable

**Create test file:** `tests/data/test_schema_parser.py`

Create pytest tests that:
1. Test parse_schema_file() on academic/schema.sql
2. Verify it extracts 14 tables correctly
3. Verify it detects foreign keys
4. Test validate_schema_content() with valid and invalid SQL
5. Test that schema_index.json is valid JSON
6. Test error handling when schema.sql is missing
7. Test that parsed table names match expected tables for academic database: ["author", "conference", "domain", "domain_author", "domain_conference", "journal", "domain_journal", "keyword", "domain_keyword", "publication", "domain_publication", "organization", "publication_keyword", "writes", "cite"]

### Expected Output Structure:

**Copied schema files:**
```
data/processed/schemas/
├── academic_schema.sql         (copy of original)
├── concert_singer_schema.sql
├── pets_1_schema.sql
└── ...
```

**schema_index.json:**
```json
{
  "academic": {
    "schema_file": "data/processed/schemas/academic_schema.sql",
    "source_file": "data/raw/spider/database/academic/schema.sql",
    "num_tables": 14,
    "tables": [
      "author", "conference", "domain", "domain_author",
      "domain_conference", "journal", "domain_journal",
      "keyword", "domain_keyword", "publication",
      "domain_publication", "organization",
      "publication_keyword", "writes", "cite"
    ],
    "has_foreign_keys": true,
    "has_primary_keys": true,
    "parsed_timestamp": "2024-12-09T10:30:00"
  }
}
```

### Testing Requirements:

After creation:
1. Running `./scripts/parse_schemas.sh` successfully parses all schemas
2. `data/processed/schemas/` contains 200+ copied schema files
3. `schema_index.json` contains metadata for all databases
4. Running `pytest tests/data/test_schema_parser.py -v` - all tests pass
5. Validation shows high match rate with tables.json

### Commit Message:
"feat(data): Add schema parser for existing Spider schema files - Module 1.2"

---

## ✅ MODULE 1.2 COMPLETION CHECKLIST

**After running the AI IDE prompt, verify the following:**

### Files Created:
- [ ] `data/schema_parser.py` exists
- [ ] `scripts/parse_schemas.sh` exists and is executable
- [ ] `tests/data/test_schema_parser.py` exists

### Schema Processing:
- [ ] `data/processed/schemas/` contains 200+ .sql files (copied from originals)
- [ ] Each copied schema maintains original formatting
- [ ] `schema_index.json` exists with all database entries

### Parsing Validation:
- [ ] Open `data/processed/schemas/academic_schema.sql`
- [ ] Should be identical to source file
- [ ] Contains PRAGMA foreign_keys statement
- [ ] Contains 14 CREATE TABLE statements
- [ ] Has foreign key constraints visible

### Index Validation:
- [ ] Open `schema_index.json`
- [ ] "academic" entry shows 14 tables
- [ ] Table names list matches what you see in schema.sql
- [ ] "has_foreign_keys" is true for academic
- [ ] Spot-check 5 random databases: verify table counts are reasonable (2-15 tables typically)

### Functional Tests:
- [ ] Run `pytest tests/data/test_schema_parser.py -v`
- [ ] All 7+ tests pass
- [ ] Test specifically verifies academic database has 15 tables (author, conference, domain, etc.)
- [ ] Foreign key detection test passes

### Cross-Reference Check:
- [ ] Validation report shows that parsed databases match tables.json
- [ ] All databases in tables.json have corresponding schema files
- [ ] Table name lists approximately match (minor differences okay)

### Understanding Check (Learning):
- [ ] You understand why Spider provides pre-written schemas (consistency, correctness)
- [ ] You know what PRAGMA foreign_keys = ON means (enables foreign key enforcement in SQLite)
- [ ] You can identify foreign key syntax: `foreign key("aid") references author("aid")`
- [ ] You understand why we copy schemas to processed/ (organization, preprocessing)
- [ ] You see the difference between simple schemas (2-3 tables) vs complex (10+ tables with relationships)

### Data Quality Insights:
- [ ] Count how many databases have <5 tables (simple schemas)
- [ ] Count how many databases have >10 tables (complex schemas)
- [ ] Identify at least 3 databases with foreign keys (relationships between tables)
- [ ] Note: Complex schemas test JOIN capabilities better

### Git Verification:
- [ ] Original schema files in `data/raw/spider/` are NOT tracked (large dataset)
- [ ] Processed schemas in `data/processed/schemas/` are NOT tracked (.gitignore)
- [ ] Only code files are staged (schema_parser.py, tests, scripts)
- [ ] schema_index.json IS tracked (small metadata file, useful)

### What You Should Have Learned:
1. **Spider Dataset Structure**: Each database has both .sqlite file AND schema.sql file
2. **Schema Format**: SQL CREATE TABLE syntax with constraints (primary keys, foreign keys)
3. **Parsing vs Extraction**: Sometimes data is already formatted, just need to parse and organize
4. **Foreign Keys**: Define relationships between tables (crucial for JOINs)
5. **Metadata Extraction**: Extracting structure information (table names, counts) without executing SQL

---

**TROUBLESHOOTING (if checks fail):**

❌ **Missing schema.sql files:**
- Check if Spider download was complete
- Some databases might not have schema.sql (use SQLite extraction as fallback)
- Should have 146+ schema files (not all 200 directories have schemas)

❌ **Parse errors:**
- Check regex for table name extraction
- Some schemas use double quotes, some single quotes, some none
- Regex should handle: `CREATE TABLE "name"`, `CREATE TABLE 'name'`, `CREATE TABLE name`

❌ **Table count mismatches with tables.json:**
- Minor differences are okay (views vs tables)
- Focus on getting 90%+ match rate
- Some databases in tables.json might not have schema.sql files

❌ **Tests fail on academic database:**
- Verify the 15 table names list in test matches actual schema.sql
- Check if foreign key detection regex works

---

**SAMPLE OUTPUT YOU SHOULD SEE:**

When running `./scripts/parse_schemas.sh`:
```
Parsing Spider schema files...
Processing: 100%|████████████| 200/200 [00:03<00:00, 66.67db/s]

✅ Parsing Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total directories found: 200
Schema files found: 166
Successfully parsed: 166
Missing schema.sql: 34
Parse errors: 0
Schemas copied to: data/processed/schemas/
Index created: schema_index.json
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Schema Statistics:
- Databases with <5 tables: 45
- Databases with 5-10 tables: 89
- Databases with >10 tables: 32
- Databases with foreign keys: 142

Running cross-reference validation...
✅ 164/166 databases match tables.json
⚠️  2 minor discrepancies found (see details above)
```