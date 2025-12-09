#!/usr/bin/env python3
"""
Verify Spider dataset integrity and print statistics.

This script validates the downloaded Spider dataset by:
1. Checking all required files exist
2. Validating JSON structure
3. Counting examples and databases
4. Printing sample data
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter

# Project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
SPIDER_DIR = PROJECT_DIR / "data" / "raw" / "spider"

# Required files
REQUIRED_FILES = [
    "train_spider.json",
    "train_others.json", 
    "dev.json",
    "tables.json",
]

# Required fields in each example
REQUIRED_FIELDS = ["question", "query", "db_id"]


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n📌 {title}")
    print("-" * 40)


def load_json(filepath: Path) -> list | dict | None:
    """Load and parse a JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error in {filepath.name}: {e}")
        return None
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None


def check_required_files() -> tuple[bool, dict]:
    """Check if all required files exist."""
    print_section("Checking Required Files")
    
    all_exist = True
    file_info = {}
    
    for filename in REQUIRED_FILES:
        filepath = SPIDER_DIR / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename} ({size_mb:.2f} MB)")
            file_info[filename] = {"exists": True, "size_mb": size_mb}
        else:
            print(f"❌ Missing: {filename}")
            file_info[filename] = {"exists": False}
            all_exist = False
    
    # Check database directory
    db_dir = SPIDER_DIR / "database"
    if db_dir.exists() and db_dir.is_dir():
        db_count = len([d for d in db_dir.iterdir() if d.is_dir()])
        print(f"✅ database/ ({db_count} directories)")
        file_info["database"] = {"exists": True, "count": db_count}
    else:
        print(f"❌ Missing: database/")
        file_info["database"] = {"exists": False}
        all_exist = False
    
    return all_exist, file_info


def validate_json_structure(data: list, filename: str) -> tuple[bool, int]:
    """Validate that JSON data has required fields."""
    print_section(f"Validating {filename}")
    
    if not isinstance(data, list):
        print(f"❌ Expected list, got {type(data).__name__}")
        return False, 0
    
    valid_count = 0
    invalid_examples = []
    
    for i, example in enumerate(data):
        missing_fields = [f for f in REQUIRED_FIELDS if f not in example]
        if missing_fields:
            if len(invalid_examples) < 3:  # Only store first 3 errors
                invalid_examples.append((i, missing_fields))
        else:
            valid_count += 1
    
    if invalid_examples:
        print(f"⚠️  Found {len(data) - valid_count} examples with missing fields:")
        for idx, fields in invalid_examples:
            print(f"   Example {idx}: missing {fields}")
    
    if valid_count == len(data):
        print(f"✅ All {valid_count} examples have required fields")
        return True, valid_count
    else:
        print(f"⚠️  {valid_count}/{len(data)} examples are valid")
        return valid_count > 0, valid_count


def analyze_sql_types(data: list) -> Counter:
    """Analyze SQL query types in the dataset."""
    sql_keywords = Counter()
    
    keywords_to_check = [
        "SELECT", "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN",
        "GROUP BY", "ORDER BY", "HAVING", "WHERE", "DISTINCT",
        "COUNT", "SUM", "AVG", "MAX", "MIN", "LIMIT",
        "UNION", "INTERSECT", "EXCEPT", "LIKE", "IN", "EXISTS",
        "BETWEEN", "CASE", "SUBQUERY"
    ]
    
    for example in data:
        query = example.get("query", "").upper()
        for keyword in keywords_to_check:
            if keyword in query:
                sql_keywords[keyword] += 1
        
        # Check for subqueries (nested SELECT)
        if query.count("SELECT") > 1:
            sql_keywords["SUBQUERY"] += 1
    
    return sql_keywords


def check_databases() -> tuple[bool, int, int]:
    """Check database directories and SQLite files."""
    print_section("Checking Databases")
    
    db_dir = SPIDER_DIR / "database"
    if not db_dir.exists():
        print("❌ Database directory not found")
        return False, 0, 0
    
    db_dirs = [d for d in db_dir.iterdir() if d.is_dir()]
    total_dbs = len(db_dirs)
    
    dbs_with_sqlite = 0
    missing_sqlite = []
    
    for db_path in db_dirs:
        sqlite_files = list(db_path.glob("*.sqlite"))
        if sqlite_files:
            dbs_with_sqlite += 1
        else:
            if len(missing_sqlite) < 5:
                missing_sqlite.append(db_path.name)
    
    print(f"   Total database directories: {total_dbs}")
    print(f"   Directories with .sqlite files: {dbs_with_sqlite}")
    
    if missing_sqlite:
        print(f"   ⚠️  Some directories missing .sqlite: {missing_sqlite[:5]}...")
    
    # List some example databases
    print(f"\n   Sample databases:")
    for db_path in sorted(db_dirs)[:5]:
        sqlite_files = list(db_path.glob("*.sqlite"))
        sqlite_info = f"({sqlite_files[0].name})" if sqlite_files else "(no .sqlite)"
        print(f"   - {db_path.name} {sqlite_info}")
    
    if total_dbs > 5:
        print(f"   ... and {total_dbs - 5} more")
    
    return dbs_with_sqlite >= 140, total_dbs, dbs_with_sqlite


def print_sample_example(data: list, dataset_name: str) -> None:
    """Print a sample example in readable format."""
    print_section(f"Sample Example from {dataset_name}")
    
    if not data:
        print("❌ No data available")
        return
    
    example = data[0]
    
    print(f"📝 Question: {example.get('question', 'N/A')}")
    print(f"💾 Database: {example.get('db_id', 'N/A')}")
    print(f"🔍 SQL Query:")
    print(f"   {example.get('query', 'N/A')}")
    
    # Print additional fields if present
    if "question_toks" in example:
        print(f"📊 Tokenized: {' '.join(example['question_toks'][:10])}...")


def print_statistics(train_data: list, train_others: list, dev_data: list, 
                     db_count: int, sql_types: Counter) -> None:
    """Print comprehensive statistics."""
    print_header("Dataset Statistics")
    
    train_count = len(train_data) if train_data else 0
    others_count = len(train_others) if train_others else 0
    dev_count = len(dev_data) if dev_data else 0
    total = train_count + others_count + dev_count
    
    print(f"""
📊 Example Counts:
   - train_spider.json:  {train_count:,} examples
   - train_others.json:  {others_count:,} examples
   - dev.json:           {dev_count:,} examples
   ─────────────────────────────────
   Total:                {total:,} examples

📁 Database Count: {db_count} databases

🔍 SQL Query Types (in training data):""")
    
    for keyword, count in sql_types.most_common(15):
        pct = (count / train_count * 100) if train_count > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"   {keyword:12} {count:5,} ({pct:5.1f}%) {bar}")


def main() -> int:
    """Main verification function."""
    print_header("Spider Dataset Verification")
    print(f"Dataset path: {SPIDER_DIR}")
    
    errors = 0
    
    # Check if Spider directory exists
    if not SPIDER_DIR.exists():
        print(f"\n❌ Spider dataset directory not found: {SPIDER_DIR}")
        print("   Run ./scripts/download_spider.sh first")
        return 1
    
    # Check required files
    files_ok, file_info = check_required_files()
    if not files_ok:
        errors += 1
    
    # Load and validate JSON files
    train_data = load_json(SPIDER_DIR / "train_spider.json")
    train_others = load_json(SPIDER_DIR / "train_others.json")
    dev_data = load_json(SPIDER_DIR / "dev.json")
    
    if train_data:
        valid, count = validate_json_structure(train_data, "train_spider.json")
        if not valid:
            errors += 1
    else:
        errors += 1
    
    if dev_data:
        valid, count = validate_json_structure(dev_data, "dev.json")
        if not valid:
            errors += 1
    else:
        errors += 1
    
    # Check databases
    db_ok, total_dbs, dbs_with_sqlite = check_databases()
    if not db_ok:
        errors += 1
    
    # Analyze SQL types
    sql_types = Counter()
    if train_data:
        sql_types = analyze_sql_types(train_data)
    
    # Print sample example
    if train_data:
        print_sample_example(train_data, "train_spider.json")
    
    # Print statistics
    print_statistics(train_data, train_others, dev_data, total_dbs, sql_types)
    
    # Final status
    print_header("Verification Result")
    
    if errors == 0:
        print("✅ All checks passed! Spider dataset is ready for use.")
        print("\nNext steps:")
        print("  1. Explore the data: python -c \"import json; print(json.load(open('data/raw/spider/train_spider.json'))[0])\"")
        print("  2. Proceed to Module 1.2: Data preprocessing")
        return 0
    else:
        print(f"❌ Verification failed with {errors} error(s)")
        print("\nTroubleshooting:")
        print("  1. Re-run ./scripts/download_spider.sh")
        print("  2. Check disk space")
        print("  3. Try manual download from https://yale-lily.github.io/spider")
        return 1


if __name__ == "__main__":
    sys.exit(main())
