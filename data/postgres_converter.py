#!/usr/bin/env python3
"""
SQLite to PostgreSQL Converter.

This module transforms SQLite SQL syntax to PostgreSQL-compatible syntax
for both queries and schemas in the Mistral training data.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

# Input files
TRAIN_MISTRAL_JSONL = PROCESSED_DIR / "train_mistral.jsonl"
DEV_MISTRAL_JSONL = PROCESSED_DIR / "dev_mistral.jsonl"

# Output files
TRAIN_POSTGRES_JSONL = PROCESSED_DIR / "train_postgres.jsonl"
DEV_POSTGRES_JSONL = PROCESSED_DIR / "dev_postgres.jsonl"
CONVERSION_STATS_JSON = PROCESSED_DIR / "postgres_conversion_stats.json"


class ConversionStats:
    """Track conversion statistics."""
    
    def __init__(self):
        self.autoincrement_conversions = 0
        self.pragma_removals = 0
        self.datetime_now_conversions = 0
        self.date_now_conversions = 0
        self.time_now_conversions = 0
        self.strftime_conversions = 0
        self.substr_conversions = 0
        self.blob_conversions = 0
        self.real_conversions = 0
        self.concatenation_found = 0
        self.begin_transaction_removals = 0
        self.commit_removals = 0
        self.total_examples = 0
        self.examples_with_conversions = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "examples_with_conversions": self.examples_with_conversions,
            "schema_conversions": {
                "autoincrement_to_serial": self.autoincrement_conversions,
                "pragma_removals": self.pragma_removals,
                "begin_transaction_removals": self.begin_transaction_removals,
                "commit_removals": self.commit_removals,
                "blob_to_bytea": self.blob_conversions,
                "real_to_numeric": self.real_conversions,
            },
            "query_conversions": {
                "datetime_now": self.datetime_now_conversions,
                "date_now": self.date_now_conversions,
                "time_now": self.time_now_conversions,
                "strftime": self.strftime_conversions,
                "substr": self.substr_conversions,
            },
            "patterns_found": {
                "concatenation_operators": self.concatenation_found,
            },
        }


# Global stats tracker
stats = ConversionStats()


def convert_sql_query(sql_query: str) -> str:
    """
    Convert SQLite SQL query to PostgreSQL syntax.
    
    Args:
        sql_query: SQLite SQL query string
        
    Returns:
        PostgreSQL-compatible SQL string
    """
    global stats
    
    if not sql_query:
        return sql_query
    
    converted = sql_query
    
    # Track if any conversion happened
    original = converted
    
    # Date/Time function conversions
    # DATETIME('now') → CURRENT_TIMESTAMP
    datetime_now_pattern = re.compile(r"DATETIME\s*\(\s*['\"]now['\"]\s*\)", re.IGNORECASE)
    if datetime_now_pattern.search(converted):
        stats.datetime_now_conversions += 1
        converted = datetime_now_pattern.sub("CURRENT_TIMESTAMP", converted)
    
    # DATETIME('now', '-N days') → CURRENT_TIMESTAMP - INTERVAL 'N days'
    datetime_offset_pattern = re.compile(
        r"DATETIME\s*\(\s*['\"]now['\"]\s*,\s*['\"]([+-]?\d+)\s*(day|days|hour|hours|minute|minutes|second|seconds)['\"]\s*\)",
        re.IGNORECASE
    )
    match = datetime_offset_pattern.search(converted)
    if match:
        stats.datetime_now_conversions += 1
        offset = match.group(1)
        unit = match.group(2).upper()
        if not unit.endswith('S'):
            unit += 'S'
        if offset.startswith('-'):
            converted = datetime_offset_pattern.sub(
                f"CURRENT_TIMESTAMP - INTERVAL '{offset[1:]} {unit}'", converted
            )
        else:
            offset_val = offset.lstrip('+')
            converted = datetime_offset_pattern.sub(
                f"CURRENT_TIMESTAMP + INTERVAL '{offset_val} {unit}'", converted
            )
    
    # DATE('now') → CURRENT_DATE
    date_now_pattern = re.compile(r"DATE\s*\(\s*['\"]now['\"]\s*\)", re.IGNORECASE)
    if date_now_pattern.search(converted):
        stats.date_now_conversions += 1
        converted = date_now_pattern.sub("CURRENT_DATE", converted)
    
    # TIME('now') → CURRENT_TIME
    time_now_pattern = re.compile(r"TIME\s*\(\s*['\"]now['\"]\s*\)", re.IGNORECASE)
    if time_now_pattern.search(converted):
        stats.time_now_conversions += 1
        converted = time_now_pattern.sub("CURRENT_TIME", converted)
    
    # strftime('%Y-%m-%d', date) → TO_CHAR(date, 'YYYY-MM-DD')
    # Common format conversions
    strftime_pattern = re.compile(
        r"strftime\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*([^)]+)\)",
        re.IGNORECASE
    )
    match = strftime_pattern.search(converted)
    if match:
        stats.strftime_conversions += 1
        sqlite_format = match.group(1)
        date_expr = match.group(2).strip()
        
        # Convert SQLite format to PostgreSQL format
        pg_format = sqlite_format
        pg_format = pg_format.replace('%Y', 'YYYY')
        pg_format = pg_format.replace('%m', 'MM')
        pg_format = pg_format.replace('%d', 'DD')
        pg_format = pg_format.replace('%H', 'HH24')
        pg_format = pg_format.replace('%M', 'MI')
        pg_format = pg_format.replace('%S', 'SS')
        
        converted = strftime_pattern.sub(f"TO_CHAR({date_expr}, '{pg_format}')", converted)
    
    # SUBSTR(str, start, length) → SUBSTRING(str FROM start FOR length)
    substr_pattern = re.compile(
        r"SUBSTR\s*\(\s*([^,]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)",
        re.IGNORECASE
    )
    match = substr_pattern.search(converted)
    if match:
        stats.substr_conversions += 1
        string_expr = match.group(1).strip()
        start = match.group(2)
        length = match.group(3)
        converted = substr_pattern.sub(
            f"SUBSTRING({string_expr} FROM {start} FOR {length})", converted
        )
    
    # Count concatenation operators (keep as-is, PostgreSQL supports ||)
    if '||' in converted:
        stats.concatenation_found += 1
    
    return converted


def convert_schema(schema_text: str) -> str:
    """
    Convert SQLite schema to PostgreSQL syntax.
    
    Args:
        schema_text: SQLite schema (CREATE TABLE statements)
        
    Returns:
        PostgreSQL-compatible schema string
    """
    global stats
    
    if not schema_text:
        return schema_text
    
    converted = schema_text
    lines = converted.split('\n')
    new_lines = []
    
    for line in lines:
        # Remove PRAGMA statements
        if re.match(r'^\s*PRAGMA\s+', line, re.IGNORECASE):
            stats.pragma_removals += 1
            continue
        
        # Remove BEGIN TRANSACTION
        if re.match(r'^\s*BEGIN\s+TRANSACTION\s*;?\s*$', line, re.IGNORECASE):
            stats.begin_transaction_removals += 1
            continue
        
        # Remove COMMIT
        if re.match(r'^\s*COMMIT\s*;?\s*$', line, re.IGNORECASE):
            stats.commit_removals += 1
            continue
        
        # Convert INTEGER PRIMARY KEY AUTOINCREMENT → SERIAL PRIMARY KEY
        autoincrement_pk_pattern = re.compile(
            r'INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT',
            re.IGNORECASE
        )
        if autoincrement_pk_pattern.search(line):
            stats.autoincrement_conversions += 1
            line = autoincrement_pk_pattern.sub('SERIAL PRIMARY KEY', line)
        
        # Convert INTEGER AUTOINCREMENT → SERIAL (without PRIMARY KEY)
        autoincrement_pattern = re.compile(
            r'INTEGER\s+AUTOINCREMENT',
            re.IGNORECASE
        )
        if autoincrement_pattern.search(line):
            stats.autoincrement_conversions += 1
            line = autoincrement_pattern.sub('SERIAL', line)
        
        # Convert BLOB → BYTEA
        blob_pattern = re.compile(r'\bBLOB\b', re.IGNORECASE)
        if blob_pattern.search(line):
            stats.blob_conversions += 1
            line = blob_pattern.sub('BYTEA', line)
        
        # Convert REAL → DOUBLE PRECISION (more precise than NUMERIC for floating point)
        # Only convert standalone REAL, not part of other words
        real_pattern = re.compile(r'\bREAL\b', re.IGNORECASE)
        if real_pattern.search(line):
            stats.real_conversions += 1
            line = real_pattern.sub('DOUBLE PRECISION', line)
        
        # Convert DATETIME('now') in DEFAULT clauses → CURRENT_TIMESTAMP
        datetime_default_pattern = re.compile(
            r"DEFAULT\s+DATETIME\s*\(\s*['\"]now['\"]\s*\)",
            re.IGNORECASE
        )
        if datetime_default_pattern.search(line):
            line = datetime_default_pattern.sub('DEFAULT CURRENT_TIMESTAMP', line)
        
        # Convert DATETIME type → TIMESTAMP
        datetime_type_pattern = re.compile(r'\bDATETIME\b', re.IGNORECASE)
        line = datetime_type_pattern.sub('TIMESTAMP', line)
        
        new_lines.append(line)
    
    # Remove empty lines at start/end and excessive blank lines
    converted = '\n'.join(new_lines)
    converted = re.sub(r'\n{3,}', '\n\n', converted)
    converted = converted.strip()
    
    return converted


def detect_conversion_patterns(sql_query: str) -> dict[str, bool]:
    """
    Identify which SQLite patterns are present in a query.
    
    Args:
        sql_query: SQL query string
        
    Returns:
        Dictionary of pattern names to boolean presence
    """
    patterns = {
        "concatenation": '||' in sql_query,
        "datetime_now": bool(re.search(r"DATETIME\s*\(\s*['\"]now", sql_query, re.IGNORECASE)),
        "date_now": bool(re.search(r"DATE\s*\(\s*['\"]now", sql_query, re.IGNORECASE)),
        "time_now": bool(re.search(r"TIME\s*\(\s*['\"]now", sql_query, re.IGNORECASE)),
        "strftime": bool(re.search(r"strftime\s*\(", sql_query, re.IGNORECASE)),
        "substr": bool(re.search(r"SUBSTR\s*\(", sql_query, re.IGNORECASE)),
        "autoincrement": bool(re.search(r"AUTOINCREMENT", sql_query, re.IGNORECASE)),
        "pragma": bool(re.search(r"PRAGMA\s+", sql_query, re.IGNORECASE)),
    }
    return patterns


def validate_postgres_syntax(sql_query: str) -> tuple[bool, list[str]]:
    """
    Basic validation for PostgreSQL syntax.
    
    Args:
        sql_query: SQL query string
        
    Returns:
        Tuple of (is_valid, list of issues found)
    """
    issues = []
    
    # Check for SQLite-specific syntax that wasn't converted
    sqlite_patterns = [
        (r"DATETIME\s*\(\s*['\"]now", "DATETIME('now') not converted"),
        (r"DATE\s*\(\s*['\"]now", "DATE('now') not converted"),
        (r"TIME\s*\(\s*['\"]now", "TIME('now') not converted"),
        (r"\bAUTOINCREMENT\b", "AUTOINCREMENT not converted"),
        (r"^\s*PRAGMA\s+", "PRAGMA statement not removed"),
    ]
    
    for pattern, message in sqlite_patterns:
        if re.search(pattern, sql_query, re.IGNORECASE):
            issues.append(message)
    
    return len(issues) == 0, issues


def convert_example(example: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a single Mistral example from SQLite to PostgreSQL.
    
    Args:
        example: Mistral format example dictionary
        
    Returns:
        Converted example dictionary
    """
    global stats
    
    # Deep copy to avoid modifying original
    converted = {
        "messages": [
            dict(msg) for msg in example["messages"]
        ]
    }
    
    # Track if any conversion happened
    had_conversion = False
    
    # Convert assistant's SQL query
    original_sql = converted["messages"][2]["content"]
    converted_sql = convert_sql_query(original_sql)
    if converted_sql != original_sql:
        had_conversion = True
    converted["messages"][2]["content"] = converted_sql
    
    # Convert schema in user message
    user_content = converted["messages"][1]["content"]
    
    # Extract schema section
    schema_start = user_content.find("Schema:\n")
    question_start = user_content.find("\n\nQuestion:")
    
    if schema_start != -1 and question_start != -1:
        schema_start += len("Schema:\n")
        original_schema = user_content[schema_start:question_start]
        converted_schema = convert_schema(original_schema)
        
        if converted_schema != original_schema:
            had_conversion = True
        
        # Rebuild user content with converted schema
        converted["messages"][1]["content"] = (
            user_content[:schema_start] +
            converted_schema +
            user_content[question_start:]
        )
    
    if had_conversion:
        stats.examples_with_conversions += 1
    
    return converted


def convert_jsonl_file(
    input_path: Path,
    output_path: Path,
    description: str = "Processing",
    verbose: bool = True
) -> int:
    """
    Convert a JSONL file from SQLite to PostgreSQL syntax.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        description: Description for progress bar
        verbose: Whether to show progress bar
        
    Returns:
        Number of examples converted
    """
    global stats
    
    # Count lines first
    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    
    count = 0
    with open(input_path, "r", encoding="utf-8") as in_f, \
         open(output_path, "w", encoding="utf-8") as out_f:
        
        iterator = tqdm(in_f, total=total_lines, desc=description, disable=not verbose)
        
        for line in iterator:
            example = json.loads(line)
            stats.total_examples += 1
            
            converted = convert_example(example)
            out_f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    
    return count


def convert_to_postgres(verbose: bool = True) -> dict[str, Any]:
    """
    Convert all Mistral JSONL files from SQLite to PostgreSQL syntax.
    
    Args:
        verbose: Whether to show progress
        
    Returns:
        Dictionary with conversion results
    """
    global stats
    stats = ConversionStats()  # Reset stats
    
    results = {
        "train_count": 0,
        "dev_count": 0,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    # Convert training data
    if TRAIN_MISTRAL_JSONL.exists():
        if verbose:
            print("\n📝 Converting training examples...")
        results["train_count"] = convert_jsonl_file(
            TRAIN_MISTRAL_JSONL,
            TRAIN_POSTGRES_JSONL,
            description="Processing train_mistral.jsonl",
            verbose=verbose
        )
    else:
        results["train_error"] = f"File not found: {TRAIN_MISTRAL_JSONL}"
    
    # Convert dev data
    if DEV_MISTRAL_JSONL.exists():
        if verbose:
            print("\n📝 Converting validation examples...")
        results["dev_count"] = convert_jsonl_file(
            DEV_MISTRAL_JSONL,
            DEV_POSTGRES_JSONL,
            description="Processing dev_mistral.jsonl",
            verbose=verbose
        )
    else:
        results["dev_error"] = f"File not found: {DEV_MISTRAL_JSONL}"
    
    return results


def analyze_conversions(verbose: bool = True) -> dict[str, Any]:
    """
    Analyze and save conversion statistics.
    
    Args:
        verbose: Whether to print statistics
        
    Returns:
        Dictionary with conversion statistics
    """
    global stats
    
    stats_dict = stats.to_dict()
    
    # Save statistics
    with open(CONVERSION_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)
    
    if verbose:
        print("\n📊 Conversion Statistics:")
        print(f"   Total examples: {stats.total_examples:,}")
        print(f"   Examples with conversions: {stats.examples_with_conversions:,}")
        print("\n   Schema conversions:")
        print(f"   - AUTOINCREMENT → SERIAL: {stats.autoincrement_conversions:,}")
        print(f"   - PRAGMA removals: {stats.pragma_removals:,}")
        print(f"   - BEGIN TRANSACTION removals: {stats.begin_transaction_removals:,}")
        print(f"   - COMMIT removals: {stats.commit_removals:,}")
        print(f"   - BLOB → BYTEA: {stats.blob_conversions:,}")
        print(f"   - REAL → DOUBLE PRECISION: {stats.real_conversions:,}")
        print("\n   Query conversions:")
        print(f"   - DATETIME('now') → CURRENT_TIMESTAMP: {stats.datetime_now_conversions:,}")
        print(f"   - DATE('now') → CURRENT_DATE: {stats.date_now_conversions:,}")
        print(f"   - strftime → TO_CHAR: {stats.strftime_conversions:,}")
        print(f"   - SUBSTR → SUBSTRING: {stats.substr_conversions:,}")
        print("\n   Patterns found (kept as-is):")
        print(f"   - Concatenation (||): {stats.concatenation_found:,}")
    
    return stats_dict


def validate_sample(n_samples: int = 100, verbose: bool = True) -> bool:
    """
    Validate a sample of converted examples.
    
    Args:
        n_samples: Number of samples to validate
        verbose: Whether to print results
        
    Returns:
        True if all samples pass validation
    """
    if not TRAIN_POSTGRES_JSONL.exists():
        if verbose:
            print("❌ Output file not found")
        return False
    
    issues_found = []
    
    with open(TRAIN_POSTGRES_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            
            example = json.loads(line)
            sql = example["messages"][2]["content"]
            schema_section = example["messages"][1]["content"]
            
            # Validate SQL query
            is_valid, sql_issues = validate_postgres_syntax(sql)
            if not is_valid:
                issues_found.extend([(i, issue) for issue in sql_issues])
            
            # Validate schema
            is_valid, schema_issues = validate_postgres_syntax(schema_section)
            if not is_valid:
                issues_found.extend([(i, issue) for issue in schema_issues])
    
    if verbose:
        if issues_found:
            print(f"\n⚠️  Found {len(issues_found)} issues in {n_samples} samples:")
            for line_num, issue in issues_found[:10]:
                print(f"   Line {line_num}: {issue}")
            if len(issues_found) > 10:
                print(f"   ... and {len(issues_found) - 10} more")
        else:
            print(f"\n✅ All {n_samples} checked examples have valid PostgreSQL syntax")
    
    return len(issues_found) == 0


def print_sample_conversion(verbose: bool = True) -> None:
    """Print a before/after sample conversion."""
    if not TRAIN_MISTRAL_JSONL.exists() or not TRAIN_POSTGRES_JSONL.exists():
        return
    
    # Find an example with conversions
    with open(TRAIN_MISTRAL_JSONL, "r", encoding="utf-8") as before_f, \
         open(TRAIN_POSTGRES_JSONL, "r", encoding="utf-8") as after_f:
        
        for before_line, after_line in zip(before_f, after_f):
            before = json.loads(before_line)
            after = json.loads(after_line)
            
            before_sql = before["messages"][2]["content"]
            after_sql = after["messages"][2]["content"]
            
            before_schema = before["messages"][1]["content"]
            after_schema = after["messages"][1]["content"]
            
            # Look for example with visible conversion
            if before_sql != after_sql or "PRAGMA" in before_schema:
                if verbose:
                    print("\n📋 Sample Conversion:")
                    print("━" * 50)
                    
                    # Extract database name
                    db_line = before["messages"][1]["content"].split("\n")[0]
                    print(f"   {db_line}")
                    
                    if "PRAGMA" in before_schema and "PRAGMA" not in after_schema:
                        print("\n   Schema change:")
                        print("   - Before: Contains PRAGMA statement")
                        print("   - After: PRAGMA removed")
                    
                    if before_sql != after_sql:
                        print(f"\n   Query change:")
                        print(f"   - Before: {before_sql}")
                        print(f"   - After:  {after_sql}")
                    
                    print("━" * 50)
                return


def print_summary(results: dict[str, Any]) -> None:
    """Print conversion summary."""
    print("\n" + "━" * 50)
    print("✅ Conversion Complete!")
    print("━" * 50)
    
    print(f"Training examples converted: {results['train_count']:,}")
    print(f"Validation examples converted: {results['dev_count']:,}")
    
    print()
    
    # File sizes
    if TRAIN_POSTGRES_JSONL.exists():
        size_mb = TRAIN_POSTGRES_JSONL.stat().st_size / (1024 * 1024)
        print(f"Output: {TRAIN_POSTGRES_JSONL.name} ({size_mb:.1f} MB)")
    
    if DEV_POSTGRES_JSONL.exists():
        size_mb = DEV_POSTGRES_JSONL.stat().st_size / (1024 * 1024)
        print(f"Output: {DEV_POSTGRES_JSONL.name} ({size_mb:.1f} MB)")
    
    print("━" * 50)


def main() -> int:
    """Main entry point for PostgreSQL conversion."""
    print("🔄 Converting SQLite syntax to PostgreSQL...")
    
    # Convert
    results = convert_to_postgres(verbose=True)
    
    if "train_error" in results:
        print(f"❌ Error: {results['train_error']}")
        return 1
    
    # Print summary
    print_summary(results)
    
    # Analyze conversions
    analyze_conversions(verbose=True)
    
    # Validate sample
    print("\n🔍 Validating sample of converted queries...")
    validate_sample(n_samples=100, verbose=True)
    
    # Print sample conversion
    print_sample_conversion(verbose=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
