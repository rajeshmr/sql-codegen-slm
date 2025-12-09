# MODULE 1.1: Download Spider Dataset

**Objective:** Download and extract the Spider dataset (Yale NLP benchmark for text-to-SQL)

---

## PROMPT FOR AI IDE (Windsurf/Claude):

### Context
You are working on sql-codegen-slm project. The Spider dataset is a large-scale text-to-SQL benchmark from Yale containing 10,181 questions across 200 databases with complex SQL queries including JOINs, subqueries, and aggregations. We need to download this dataset and verify its structure for training our Mistral model.

### Task: Create Spider Dataset Download Script

**Location:** `scripts/download_spider.sh`

**Script Requirements:**

The script should:
1. Create `data/raw/spider/` directory if it doesn't exist
2. Download Spider dataset from official source: https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ (this is the official Spider 1.0 download link)
3. If direct download fails, provide instructions to manually download from https://yale-lily.github.io/spider
4. Extract the zip file to `data/raw/spider/`
5. Verify extraction by checking for these key files:
   - `train_spider.json` (training examples)
   - `train_others.json` (additional training data)
   - `dev.json` (development/validation set)
   - `tables.json` (database schema information)
   - `database/` directory (should contain 200+ SQLite database directories)
6. Print download statistics:
   - Total size downloaded
   - Number of databases found
   - Number of training examples (from train_spider.json)
   - Number of dev examples (from dev.json)
7. Create a summary file `data/raw/spider/download_summary.txt` with:
   - Download timestamp
   - Dataset version
   - File counts
   - Any warnings or errors
8. Handle errors gracefully with clear messages
9. Make script executable (chmod +x)

**Additional File:** `scripts/verify_spider.py`

Create a Python verification script that:
1. Loads `train_spider.json` and prints first example in readable format
2. Counts total examples in train_spider.json, train_others.json, and dev.json
3. Lists all database directories in `database/`
4. Checks if all databases have `.sqlite` files
5. Validates JSON structure (has required fields: question, query, db_id)
6. Prints summary statistics:
   - Total training examples
   - Total dev examples  
   - Total databases
   - Example question types (SELECT, JOIN, GROUP BY, etc.)
7. Exit with code 0 if all checks pass, 1 if any fail

**Update requirements.txt:**
Add these dependencies (with versions):
```
requests>=2.31.0
tqdm>=4.66.0
```

**Update README.md:**
Add a new section "Data Download" with:
- Instructions to run `./scripts/download_spider.sh`
- Expected output and file structure
- Link to Spider dataset paper/website
- Troubleshooting section for download issues

### Testing Requirements:

After creation, these should work:
1. Running `./scripts/download_spider.sh` downloads and extracts Spider dataset
2. Running `python scripts/verify_spider.py` shows dataset statistics and passes all checks
3. `data/raw/spider/database/` contains 200+ directories
4. `data/raw/spider/train_spider.json` is valid JSON with SQL examples
5. No errors in download_summary.txt

### Expected File Structure After Download:
```
data/raw/spider/
├── train_spider.json          (~8,659 examples)
├── train_others.json          (~1,659 examples)  
├── dev.json                   (~1,034 examples)
├── tables.json                (schema definitions)
├── database/                  (200+ SQLite databases)
│   ├── concert_singer/
│   │   └── concert_singer.sqlite
│   ├── pets_1/
│   │   └── pets_1.sqlite
│   └── ...
└── download_summary.txt       (download verification)
```

### Commit Message:
"feat(data): Add Spider dataset download and verification scripts - Module 1.1"

---

## ✅ MODULE 1.1 COMPLETION CHECKLIST

**After running the AI IDE prompt, verify the following:**

### Files Created:
- [ ] `scripts/download_spider.sh` exists and is executable
- [ ] `scripts/verify_spider.py` exists
- [ ] `requirements.txt` has `requests` and `tqdm` dependencies
- [ ] README.md has "Data Download" section

### Download Verification:
- [ ] `data/raw/spider/` directory exists
- [ ] `train_spider.json` file exists (should be ~10MB)
- [ ] `dev.json` file exists
- [ ] `tables.json` file exists
- [ ] `database/` directory exists with 200+ subdirectories
- [ ] `download_summary.txt` shows successful download

### Data Validation:
- [ ] Run `python scripts/verify_spider.py` - should show statistics
- [ ] Training examples count: ~8,659 (from train_spider.json)
- [ ] Dev examples count: ~1,034 (from dev.json)
- [ ] Database count: 200+ directories
- [ ] Sample example printed shows: question, SQL query, database name
- [ ] No JSON parsing errors

### Functional Tests:
- [ ] Can open `data/raw/spider/train_spider.json` in text editor without errors
- [ ] Sample SQLite database can be opened: `sqlite3 data/raw/spider/database/concert_singer/concert_singer.sqlite`
- [ ] Running `.schema` in sqlite3 shows table definitions

### Understanding Check (Learning):
- [ ] You understand what Spider dataset contains (questions + SQL + databases)
- [ ] You can explain what train_spider.json structure looks like
- [ ] You know why we need 200+ databases (diverse schemas for generalization)
- [ ] You understand the difference between train_spider.json and dev.json (training vs validation)

### Git Verification:
- [ ] Changes staged for commit
- [ ] Commit message follows convention
- [ ] `.gitignore` is preventing `data/raw/spider/database/*.sqlite` from being tracked (large files)

### What You Should Have Learned:
1. **Spider Dataset Structure**: Questions, SQL queries, and corresponding databases
2. **Dataset Size**: ~10K training examples across 200+ diverse database schemas
3. **Verification Importance**: Always validate downloaded data before using it
4. **Bash + Python Integration**: Shell script for download, Python for validation

---

**TROUBLESHOOTING (if checks fail):**

❌ **Download fails:**
- Try manual download from https://yale-lily.github.io/spider
- Check internet connection
- Verify Google Drive link is still active

❌ **JSON parsing errors:**
- Re-download the dataset (file may be corrupted)
- Check if extraction completed fully

❌ **Database count wrong:**
- Verify extraction completed (check disk space)
- Some subdirectories might not have .sqlite files (that's okay if 146+ have them)