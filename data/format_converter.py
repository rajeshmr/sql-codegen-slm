#!/usr/bin/env python3
"""
Format Converter for Spider to Mistral Instruction Format.

This module transforms Spider JSON data into Mistral instruction tuning format (JSONL)
with schema context embedded in the user message.
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
RAW_SPIDER_DIR = PROJECT_DIR / "data" / "raw" / "spider"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SCHEMAS_DIR = PROCESSED_DIR / "schemas"

# Input files
TRAIN_SPIDER_JSON = RAW_SPIDER_DIR / "train_spider.json"
TRAIN_OTHERS_JSON = RAW_SPIDER_DIR / "train_others.json"
DEV_JSON = RAW_SPIDER_DIR / "dev.json"
SCHEMA_INDEX_JSON = SCHEMAS_DIR / "schema_index.json"

# Output files
TRAIN_MISTRAL_JSONL = PROCESSED_DIR / "train_mistral.jsonl"
DEV_MISTRAL_JSONL = PROCESSED_DIR / "dev_mistral.jsonl"
DATASET_STATS_JSON = PROCESSED_DIR / "dataset_statistics.json"

# System prompt for Mistral
SYSTEM_PROMPT = "You are an expert PostgreSQL query generator. Given a database schema and a natural language question, generate a valid SQL query."


def clean_sql_query(sql_query: str) -> str:
    """
    Clean and normalize SQL query.
    
    Args:
        sql_query: Raw SQL query string
        
    Returns:
        Cleaned SQL query with normalized whitespace and semicolon
    """
    if not sql_query:
        return ""
    
    # Remove extra whitespace (multiple spaces, tabs, newlines)
    cleaned = " ".join(sql_query.split())
    
    # Ensure query ends with semicolon
    cleaned = cleaned.strip()
    if not cleaned.endswith(";"):
        cleaned += ";"
    
    return cleaned


def load_schema_content(db_id: str, schema_index: dict[str, Any]) -> str | None:
    """
    Load schema content for a database.
    
    Args:
        db_id: Database identifier
        schema_index: Dictionary mapping db_id to schema metadata
        
    Returns:
        Schema content as string, or None if not found
    """
    if db_id not in schema_index:
        return None
    
    schema_info = schema_index[db_id]
    schema_file = PROJECT_DIR / schema_info["schema_file"]
    
    if not schema_file.exists():
        return None
    
    try:
        return schema_file.read_text(encoding="utf-8")
    except Exception:
        return None


def format_mistral_example(
    question: str,
    sql_query: str,
    schema_content: str,
    db_id: str
) -> dict[str, Any]:
    """
    Format a single example in Mistral instruction format.
    
    Args:
        question: Natural language question
        sql_query: SQL query (answer)
        schema_content: Database schema as SQL CREATE statements
        db_id: Database identifier
        
    Returns:
        Dictionary in Mistral messages format
    """
    # Clean the SQL query
    cleaned_sql = clean_sql_query(sql_query)
    
    # Format user message with schema context
    user_content = f"""Database: {db_id}

Schema:
{schema_content}

Question: {question}

Generate the SQL query:"""
    
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": cleaned_sql
            }
        ]
    }


def convert_spider_file(
    input_path: Path,
    output_path: Path,
    schema_index: dict[str, Any],
    description: str = "Processing",
    verbose: bool = True
) -> dict[str, int]:
    """
    Convert a Spider JSON file to Mistral JSONL format.
    
    Args:
        input_path: Path to Spider JSON file
        output_path: Path to output JSONL file
        schema_index: Dictionary mapping db_id to schema metadata
        description: Description for progress bar
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with conversion statistics
    """
    stats = {
        "total": 0,
        "written": 0,
        "skipped_missing_schema": 0,
        "skipped_empty_query": 0,
    }
    
    # Load Spider examples
    with open(input_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    
    stats["total"] = len(examples)
    
    # Schema content cache to avoid re-reading files
    schema_cache: dict[str, str | None] = {}
    
    # Convert and write
    with open(output_path, "w", encoding="utf-8") as out_f:
        iterator = tqdm(examples, desc=description, disable=not verbose)
        
        for example in iterator:
            db_id = example.get("db_id", "")
            question = example.get("question", "")
            sql_query = example.get("query", "")
            
            # Skip if no query
            if not sql_query.strip():
                stats["skipped_empty_query"] += 1
                continue
            
            # Load schema (with caching)
            if db_id not in schema_cache:
                schema_cache[db_id] = load_schema_content(db_id, schema_index)
            
            schema_content = schema_cache[db_id]
            
            # Skip if schema not found
            if schema_content is None:
                stats["skipped_missing_schema"] += 1
                continue
            
            # Format as Mistral example
            mistral_example = format_mistral_example(
                question=question,
                sql_query=sql_query,
                schema_content=schema_content,
                db_id=db_id
            )
            
            # Write as single JSON line
            out_f.write(json.dumps(mistral_example, ensure_ascii=False) + "\n")
            stats["written"] += 1
    
    return stats


def convert_spider_to_mistral(verbose: bool = True) -> dict[str, Any]:
    """
    Convert all Spider examples to Mistral instruction format.
    
    Args:
        verbose: Whether to show progress
        
    Returns:
        Dictionary with overall conversion statistics
    """
    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load schema index
    if not SCHEMA_INDEX_JSON.exists():
        return {"error": f"Schema index not found: {SCHEMA_INDEX_JSON}"}
    
    with open(SCHEMA_INDEX_JSON, "r", encoding="utf-8") as f:
        schema_index = json.load(f)
    
    results = {
        "train": {},
        "dev": {},
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    # Convert training data (train_spider.json only, as specified)
    if TRAIN_SPIDER_JSON.exists():
        if verbose:
            print("\n📝 Converting training examples...")
        results["train"] = convert_spider_file(
            TRAIN_SPIDER_JSON,
            TRAIN_MISTRAL_JSONL,
            schema_index,
            description="Processing train_spider.json",
            verbose=verbose
        )
    else:
        results["train"] = {"error": f"File not found: {TRAIN_SPIDER_JSON}"}
    
    # Convert dev data
    if DEV_JSON.exists():
        if verbose:
            print("\n📝 Converting validation examples...")
        results["dev"] = convert_spider_file(
            DEV_JSON,
            DEV_MISTRAL_JSONL,
            schema_index,
            description="Processing dev.json",
            verbose=verbose
        )
    else:
        results["dev"] = {"error": f"File not found: {DEV_JSON}"}
    
    return results


def analyze_dataset(verbose: bool = True) -> dict[str, Any]:
    """
    Analyze the converted dataset for query complexity statistics.
    
    Args:
        verbose: Whether to print statistics
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "total_examples": 0,
        "simple_select": 0,  # No JOIN
        "with_join": 0,
        "with_group_by": 0,
        "with_subquery": 0,
        "with_order_by": 0,
        "with_limit": 0,
        "with_having": 0,
        "with_distinct": 0,
        "with_aggregation": 0,  # COUNT, SUM, AVG, MAX, MIN
    }
    
    if not TRAIN_MISTRAL_JSONL.exists():
        return {"error": "Training JSONL not found. Run conversion first."}
    
    # Analyze training file
    with open(TRAIN_MISTRAL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            sql = example["messages"][2]["content"].upper()
            
            stats["total_examples"] += 1
            
            # Check for various SQL features
            has_join = "JOIN" in sql
            has_group_by = "GROUP BY" in sql
            has_subquery = sql.count("SELECT") > 1
            has_order_by = "ORDER BY" in sql
            has_limit = "LIMIT" in sql
            has_having = "HAVING" in sql
            has_distinct = "DISTINCT" in sql
            has_aggregation = any(agg in sql for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("])
            
            if has_join:
                stats["with_join"] += 1
            else:
                stats["simple_select"] += 1
            
            if has_group_by:
                stats["with_group_by"] += 1
            if has_subquery:
                stats["with_subquery"] += 1
            if has_order_by:
                stats["with_order_by"] += 1
            if has_limit:
                stats["with_limit"] += 1
            if has_having:
                stats["with_having"] += 1
            if has_distinct:
                stats["with_distinct"] += 1
            if has_aggregation:
                stats["with_aggregation"] += 1
    
    # Calculate percentages
    total = stats["total_examples"]
    if total > 0:
        stats["percentages"] = {
            "simple_select": round(stats["simple_select"] / total * 100, 1),
            "with_join": round(stats["with_join"] / total * 100, 1),
            "with_group_by": round(stats["with_group_by"] / total * 100, 1),
            "with_subquery": round(stats["with_subquery"] / total * 100, 1),
            "with_order_by": round(stats["with_order_by"] / total * 100, 1),
            "with_aggregation": round(stats["with_aggregation"] / total * 100, 1),
        }
    
    # Save statistics
    with open(DATASET_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    if verbose:
        print("\n📊 Dataset Statistics:")
        print(f"   Total examples: {stats['total_examples']:,}")
        print(f"   Simple SELECT (no JOIN): {stats['simple_select']:,} ({stats['percentages']['simple_select']}%)")
        print(f"   With JOIN: {stats['with_join']:,} ({stats['percentages']['with_join']}%)")
        print(f"   With GROUP BY: {stats['with_group_by']:,} ({stats['percentages']['with_group_by']}%)")
        print(f"   With subquery: {stats['with_subquery']:,} ({stats['percentages']['with_subquery']}%)")
        print(f"   With ORDER BY: {stats['with_order_by']:,} ({stats['percentages']['with_order_by']}%)")
        print(f"   With aggregation: {stats['with_aggregation']:,} ({stats['percentages']['with_aggregation']}%)")
    
    return stats


def validate_conversion(verbose: bool = True) -> bool:
    """
    Validate the converted JSONL files.
    
    Args:
        verbose: Whether to print details
        
    Returns:
        True if validation passes, False otherwise
    """
    errors = []
    
    for jsonl_path in [TRAIN_MISTRAL_JSONL, DEV_MISTRAL_JSONL]:
        if not jsonl_path.exists():
            errors.append(f"File not found: {jsonl_path}")
            continue
        
        # Check first 10 examples
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"{jsonl_path.name} line {i+1}: Invalid JSON - {e}")
                    continue
                
                # Check structure
                if "messages" not in example:
                    errors.append(f"{jsonl_path.name} line {i+1}: Missing 'messages' field")
                    continue
                
                messages = example["messages"]
                if len(messages) != 3:
                    errors.append(f"{jsonl_path.name} line {i+1}: Expected 3 messages, got {len(messages)}")
                    continue
                
                # Check roles
                expected_roles = ["system", "user", "assistant"]
                for j, (msg, expected_role) in enumerate(zip(messages, expected_roles)):
                    if msg.get("role") != expected_role:
                        errors.append(f"{jsonl_path.name} line {i+1}: Message {j} has role '{msg.get('role')}', expected '{expected_role}'")
                    if "content" not in msg:
                        errors.append(f"{jsonl_path.name} line {i+1}: Message {j} missing 'content'")
                
                # Check user message has schema
                user_content = messages[1].get("content", "")
                if "Schema:" not in user_content:
                    errors.append(f"{jsonl_path.name} line {i+1}: User message missing schema")
                if "CREATE TABLE" not in user_content:
                    errors.append(f"{jsonl_path.name} line {i+1}: User message missing CREATE TABLE statements")
    
    if verbose:
        if errors:
            print("\n❌ Validation errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"   - {error}")
            if len(errors) > 10:
                print(f"   ... and {len(errors) - 10} more errors")
        else:
            print("\n✅ All examples have correct structure")
            print("✅ Schema content properly embedded")
    
    return len(errors) == 0


def print_sample_example(verbose: bool = True) -> None:
    """Print a sample formatted example."""
    if not TRAIN_MISTRAL_JSONL.exists():
        print("❌ Training JSONL not found")
        return
    
    with open(TRAIN_MISTRAL_JSONL, "r", encoding="utf-8") as f:
        example = json.loads(f.readline())
    
    messages = example["messages"]
    user_content = messages[1]["content"]
    
    # Extract parts from user message
    lines = user_content.split("\n")
    db_line = lines[0] if lines else ""
    
    # Find question
    question = ""
    for line in lines:
        if line.startswith("Question:"):
            question = line.replace("Question:", "").strip()
            break
    
    # Count CREATE TABLE statements
    schema_tables = user_content.count("CREATE TABLE")
    
    sql = messages[2]["content"]
    
    if verbose:
        print("\n📋 Sample Example:")
        print(f"   {db_line}")
        print(f"   Question: {question}")
        print(f"   Schema: [{schema_tables} CREATE TABLE statements]")
        print(f"   SQL: {sql}")


def print_summary(results: dict[str, Any]) -> None:
    """Print conversion summary."""
    print("\n" + "━" * 50)
    print("✅ Conversion Complete!")
    print("━" * 50)
    
    # Training stats
    train = results.get("train", {})
    if "error" not in train:
        total = train.get("total", 0)
        written = train.get("written", 0)
        skipped = train.get("skipped_missing_schema", 0)
        pct = (written / total * 100) if total > 0 else 0
        
        print(f"Training examples processed: {total:,}")
        print(f"Training examples written:   {written:,} ({pct:.1f}%)")
        print(f"Skipped (missing schema):    {skipped:,}")
    
    print()
    
    # Dev stats
    dev = results.get("dev", {})
    if "error" not in dev:
        total = dev.get("total", 0)
        written = dev.get("written", 0)
        skipped = dev.get("skipped_missing_schema", 0)
        pct = (written / total * 100) if total > 0 else 0
        
        print(f"Validation examples processed: {total:,}")
        print(f"Validation examples written:   {written:,} ({pct:.1f}%)")
        print(f"Skipped (missing schema):      {skipped:,}")
    
    print()
    
    # File sizes
    if TRAIN_MISTRAL_JSONL.exists():
        size_mb = TRAIN_MISTRAL_JSONL.stat().st_size / (1024 * 1024)
        print(f"Output: {TRAIN_MISTRAL_JSONL.name} ({size_mb:.1f} MB)")
    
    if DEV_MISTRAL_JSONL.exists():
        size_mb = DEV_MISTRAL_JSONL.stat().st_size / (1024 * 1024)
        print(f"Output: {DEV_MISTRAL_JSONL.name} ({size_mb:.1f} MB)")
    
    print("━" * 50)


def main() -> int:
    """Main entry point for format conversion."""
    print("🔄 Converting Spider examples to Mistral format...")
    
    # Convert
    results = convert_spider_to_mistral(verbose=True)
    
    if "error" in results.get("train", {}):
        print(f"❌ Error: {results['train']['error']}")
        return 1
    
    # Print summary
    print_summary(results)
    
    # Analyze dataset
    analyze_dataset(verbose=True)
    
    # Validate
    print("\n🔍 Validating format...")
    is_valid = validate_conversion(verbose=True)
    
    # Print sample
    print_sample_example(verbose=True)
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
