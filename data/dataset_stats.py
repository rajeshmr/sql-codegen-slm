#!/usr/bin/env python3
"""
Dataset Statistics Generator for SQL Codegen Training.

This module generates comprehensive statistics about the training,
validation, and test datasets.
"""

import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

# Input files
TRAIN_POSTGRES_JSONL = PROCESSED_DIR / "train_postgres.jsonl"
VAL_POSTGRES_JSONL = PROCESSED_DIR / "val_postgres.jsonl"
TEST_POSTGRES_JSONL = PROCESSED_DIR / "test_postgres.jsonl"

# Output file
DATASET_STATS_JSON = PROCESSED_DIR / "dataset_statistics.json"


def analyze_sql_features(sql_query: str) -> dict[str, Any]:
    """
    Extract SQL features from a query.
    
    Args:
        sql_query: SQL query string
        
    Returns:
        Dictionary of features
    """
    sql_upper = sql_query.upper()
    
    features = {
        # Statement types
        "has_select": sql_upper.startswith("SELECT") or " SELECT " in sql_upper,
        "has_insert": "INSERT" in sql_upper,
        "has_update": "UPDATE" in sql_upper,
        "has_delete": "DELETE" in sql_upper,
        
        # JOIN types
        "has_join": "JOIN" in sql_upper,
        "has_inner_join": "INNER JOIN" in sql_upper,
        "has_left_join": "LEFT JOIN" in sql_upper or "LEFT OUTER JOIN" in sql_upper,
        "has_right_join": "RIGHT JOIN" in sql_upper or "RIGHT OUTER JOIN" in sql_upper,
        "has_full_join": "FULL JOIN" in sql_upper or "FULL OUTER JOIN" in sql_upper,
        "has_cross_join": "CROSS JOIN" in sql_upper,
        
        # Clauses
        "has_where": "WHERE" in sql_upper,
        "has_group_by": "GROUP BY" in sql_upper,
        "has_having": "HAVING" in sql_upper,
        "has_order_by": "ORDER BY" in sql_upper,
        "has_limit": "LIMIT" in sql_upper,
        
        # Subqueries and set operations
        "has_subquery": sql_upper.count("SELECT") > 1,
        "has_union": "UNION" in sql_upper,
        "has_except": "EXCEPT" in sql_upper,
        "has_intersect": "INTERSECT" in sql_upper,
        
        # Aggregation functions
        "has_count": "COUNT(" in sql_upper,
        "has_sum": "SUM(" in sql_upper,
        "has_avg": "AVG(" in sql_upper,
        "has_max": "MAX(" in sql_upper,
        "has_min": "MIN(" in sql_upper,
        "has_aggregation": any(agg in sql_upper for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]),
        
        # Other features
        "has_distinct": "DISTINCT" in sql_upper,
        "has_like": "LIKE" in sql_upper,
        "has_in": " IN " in sql_upper or " IN(" in sql_upper,
        "has_between": "BETWEEN" in sql_upper,
        "has_case": "CASE " in sql_upper,
        "has_null_check": "IS NULL" in sql_upper or "IS NOT NULL" in sql_upper,
    }
    
    # Count specific elements
    features["join_count"] = len(re.findall(r'\bJOIN\b', sql_upper))
    features["subquery_count"] = max(0, sql_upper.count("SELECT") - 1)
    features["table_count"] = len(re.findall(r'\bFROM\b', sql_upper)) + features["join_count"]
    
    return features


def calculate_token_stats(examples: list[dict[str, Any]], verbose: bool = False) -> dict[str, Any]:
    """
    Calculate token/character statistics for examples.
    
    Args:
        examples: List of example dictionaries
        verbose: Whether to show progress
        
    Returns:
        Statistics dictionary
    """
    schema_lengths = []
    question_lengths = []
    sql_lengths = []
    schema_tokens = []
    question_tokens = []
    sql_tokens = []
    
    iterator = tqdm(examples, desc="Analyzing tokens", disable=not verbose)
    
    for example in iterator:
        user_content = example["messages"][1]["content"]
        sql = example["messages"][2]["content"]
        
        # Extract schema and question from user content
        schema_start = user_content.find("Schema:\n")
        question_start = user_content.find("\n\nQuestion:")
        
        if schema_start != -1 and question_start != -1:
            schema = user_content[schema_start + len("Schema:\n"):question_start]
            question_end = user_content.find("\n\nGenerate the SQL query:")
            question = user_content[question_start + len("\n\nQuestion:"):question_end].strip()
        else:
            schema = ""
            question = ""
        
        # Character lengths
        schema_lengths.append(len(schema))
        question_lengths.append(len(question))
        sql_lengths.append(len(sql))
        
        # Simple token count (split by whitespace)
        schema_tokens.append(len(schema.split()))
        question_tokens.append(len(question.split()))
        sql_tokens.append(len(sql.split()))
    
    def calc_stats(values):
        if not values:
            return {"min": 0, "max": 0, "avg": 0, "median": 0}
        sorted_vals = sorted(values)
        return {
            "min": min(values),
            "max": max(values),
            "avg": round(sum(values) / len(values), 1),
            "median": sorted_vals[len(sorted_vals) // 2],
        }
    
    return {
        "schema_chars": calc_stats(schema_lengths),
        "question_chars": calc_stats(question_lengths),
        "sql_chars": calc_stats(sql_lengths),
        "schema_tokens": calc_stats(schema_tokens),
        "question_tokens": calc_stats(question_tokens),
        "sql_tokens": calc_stats(sql_tokens),
    }


def analyze_split(
    filepath: Path,
    split_name: str,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Analyze a single split file.
    
    Args:
        filepath: Path to JSONL file
        split_name: Name of the split
        verbose: Whether to show progress
        
    Returns:
        Statistics dictionary
    """
    if not filepath.exists():
        return {"error": f"File not found: {filepath}"}
    
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    
    # Basic counts
    stats = {
        "file": str(filepath.name),
        "count": len(examples),
    }
    
    # Database diversity
    databases = set()
    
    # SQL feature counts
    feature_counts = defaultdict(int)
    
    # Complexity distribution
    complexity_dist = defaultdict(int)
    
    if verbose:
        print(f"\nAnalyzing {split_name} set ({len(examples)} examples)...")
    
    for example in examples:
        user_content = example["messages"][1]["content"]
        sql = example["messages"][2]["content"]
        
        # Extract database
        if user_content.startswith("Database:"):
            db_name = user_content.split("\n")[0].replace("Database:", "").strip()
            databases.add(db_name)
        
        # Analyze SQL features
        features = analyze_sql_features(sql)
        for feature, value in features.items():
            if isinstance(value, bool) and value:
                feature_counts[feature] += 1
        
        # Determine complexity
        sql_upper = sql.upper()
        if sql_upper.count("SELECT") > 1 or features["join_count"] >= 2:
            if sql_upper.count("SELECT") >= 2 and features["has_aggregation"]:
                complexity_dist["very_complex"] += 1
            else:
                complexity_dist["complex"] += 1
        elif features["has_join"] or features["has_group_by"]:
            complexity_dist["medium"] += 1
        else:
            complexity_dist["simple"] += 1
    
    stats["databases"] = len(databases)
    stats["complexity_distribution"] = dict(complexity_dist)
    
    # Calculate percentages for features
    total = len(examples)
    stats["sql_features"] = {
        feature: {
            "count": count,
            "percentage": round(count / total * 100, 1) if total > 0 else 0
        }
        for feature, count in sorted(feature_counts.items())
        if not feature.endswith("_count")  # Skip count fields
    }
    
    # Token statistics
    stats["token_stats"] = calculate_token_stats(examples, verbose=False)
    
    # Complexity percentages
    if total > 0:
        stats["complexity_percentages"] = {
            level: round(count / total * 100, 1)
            for level, count in complexity_dist.items()
        }
    
    return stats


def generate_comprehensive_stats(verbose: bool = True) -> dict[str, Any]:
    """
    Generate comprehensive statistics for all splits.
    
    Args:
        verbose: Whether to print progress
        
    Returns:
        Complete statistics dictionary
    """
    if verbose:
        print("\n📊 Computing dataset statistics...")
        print("━" * 50)
    
    stats = {
        "generated_at": datetime.utcnow().isoformat(),
        "splits": {},
    }
    
    # Analyze each split
    splits = [
        (TRAIN_POSTGRES_JSONL, "train"),
        (VAL_POSTGRES_JSONL, "validation"),
        (TEST_POSTGRES_JSONL, "test"),
    ]
    
    total_examples = 0
    
    for filepath, split_name in splits:
        if filepath.exists():
            split_stats = analyze_split(filepath, split_name, verbose=verbose)
            stats["splits"][split_name] = split_stats
            total_examples += split_stats.get("count", 0)
    
    stats["total_examples"] = total_examples
    
    # Save statistics
    with open(DATASET_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    if verbose:
        print(f"\n✅ Statistics saved to {DATASET_STATS_JSON}")
    
    return stats


def print_summary(stats: dict[str, Any]) -> None:
    """Print a summary of the statistics."""
    print("\n" + "━" * 50)
    print("📊 Dataset Statistics Summary")
    print("━" * 50)
    
    for split_name, split_stats in stats.get("splits", {}).items():
        if "error" in split_stats:
            continue
        
        print(f"\n{split_name.upper()} Set:")
        print(f"  Examples: {split_stats['count']:,}")
        print(f"  Databases: {split_stats['databases']}")
        
        # Complexity
        if "complexity_percentages" in split_stats:
            complex_pct = split_stats["complexity_percentages"].get("complex", 0)
            very_complex_pct = split_stats["complexity_percentages"].get("very_complex", 0)
            print(f"  Complex queries: {complex_pct + very_complex_pct:.1f}%")
        
        # Token stats
        if "token_stats" in split_stats:
            avg_sql_tokens = split_stats["token_stats"]["sql_tokens"]["avg"]
            print(f"  Avg SQL tokens: {avg_sql_tokens}")
    
    print("\n" + "━" * 50)
    print(f"Total examples: {stats.get('total_examples', 0):,}")
    print("━" * 50)


def main() -> int:
    """Main entry point for statistics generation."""
    print("📊 Generating comprehensive dataset statistics...")
    
    stats = generate_comprehensive_stats(verbose=True)
    
    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
        return 1
    
    print_summary(stats)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
