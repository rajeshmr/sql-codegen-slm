#!/usr/bin/env python3
"""
Data Split Creator for SQL Codegen Training.

This module creates train/validation/test splits from the processed data,
using stratified sampling to maintain query complexity distribution.
"""

import json
import random
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
DEV_POSTGRES_JSONL = PROCESSED_DIR / "dev_postgres.jsonl"

# Output files
VAL_POSTGRES_JSONL = PROCESSED_DIR / "val_postgres.jsonl"
TEST_POSTGRES_JSONL = PROCESSED_DIR / "test_postgres.jsonl"
SPLIT_INFO_JSON = PROCESSED_DIR / "split_info.json"

# Random seed for reproducibility
RANDOM_SEED = 42


def analyze_query_complexity(sql_query: str) -> tuple[int, dict[str, bool]]:
    """
    Determine query complexity level.
    
    Args:
        sql_query: SQL query string
        
    Returns:
        Tuple of (complexity_level, features_dict)
        Level 1: Simple - Basic SELECT, no JOIN
        Level 2: Medium - With JOIN or GROUP BY
        Level 3: Complex - With subquery or multiple JOINs
        Level 4: Very Complex - Nested subqueries, multiple aggregations
    """
    sql_upper = sql_query.upper()
    
    features = {
        "has_join": "JOIN" in sql_upper,
        "has_group_by": "GROUP BY" in sql_upper,
        "has_order_by": "ORDER BY" in sql_upper,
        "has_having": "HAVING" in sql_upper,
        "has_subquery": sql_upper.count("SELECT") > 1,
        "has_union": "UNION" in sql_upper,
        "has_except": "EXCEPT" in sql_upper,
        "has_intersect": "INTERSECT" in sql_upper,
        "has_distinct": "DISTINCT" in sql_upper,
        "has_limit": "LIMIT" in sql_upper,
        "has_aggregation": any(agg in sql_upper for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]),
    }
    
    # Count JOINs
    join_count = len(re.findall(r'\bJOIN\b', sql_upper))
    features["join_count"] = join_count
    
    # Count subqueries
    subquery_count = sql_upper.count("SELECT") - 1
    features["subquery_count"] = max(0, subquery_count)
    
    # Determine complexity level
    complexity = 1  # Simple
    
    if features["has_join"] or features["has_group_by"]:
        complexity = 2  # Medium
    
    if features["has_subquery"] or join_count >= 2:
        complexity = 3  # Complex
    
    if subquery_count >= 2 or (features["has_subquery"] and features["has_aggregation"] and features["has_group_by"]):
        complexity = 4  # Very Complex
    
    if features["has_union"] or features["has_except"] or features["has_intersect"]:
        complexity = max(complexity, 3)  # At least Complex
    
    return complexity, features


def get_complexity_label(level: int) -> str:
    """Get human-readable complexity label."""
    labels = {1: "simple", 2: "medium", 3: "complex", 4: "very_complex"}
    return labels.get(level, "unknown")


def stratified_split(
    examples: list[dict[str, Any]],
    split_ratio: float = 0.5,
    seed: int = RANDOM_SEED
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split examples while maintaining complexity distribution.
    
    Args:
        examples: List of example dictionaries
        split_ratio: Ratio for first split (default 0.5)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (first_split, second_split)
    """
    random.seed(seed)
    
    # Group examples by complexity level
    by_complexity: dict[int, list[dict[str, Any]]] = defaultdict(list)
    
    for example in examples:
        sql = example["messages"][2]["content"]
        complexity, _ = analyze_query_complexity(sql)
        by_complexity[complexity].append(example)
    
    first_split = []
    second_split = []
    
    # Split each complexity group proportionally
    for complexity, group in by_complexity.items():
        random.shuffle(group)
        split_point = int(len(group) * split_ratio)
        first_split.extend(group[:split_point])
        second_split.extend(group[split_point:])
    
    # Shuffle final splits
    random.shuffle(first_split)
    random.shuffle(second_split)
    
    return first_split, second_split


def load_jsonl(filepath: Path) -> list[dict[str, Any]]:
    """Load examples from JSONL file."""
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def save_jsonl(examples: list[dict[str, Any]], filepath: Path) -> None:
    """Save examples to JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def get_split_stats(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate statistics for a split."""
    complexity_dist = defaultdict(int)
    databases = set()
    
    for example in examples:
        sql = example["messages"][2]["content"]
        complexity, _ = analyze_query_complexity(sql)
        complexity_dist[get_complexity_label(complexity)] += 1
        
        # Extract database name from user message
        user_content = example["messages"][1]["content"]
        if user_content.startswith("Database:"):
            db_name = user_content.split("\n")[0].replace("Database:", "").strip()
            databases.add(db_name)
    
    return {
        "count": len(examples),
        "databases": len(databases),
        "complexity_distribution": dict(complexity_dist),
    }


def create_final_splits(verbose: bool = True) -> dict[str, Any]:
    """
    Create train/validation/test splits.
    
    Args:
        verbose: Whether to print progress
        
    Returns:
        Dictionary with split statistics
    """
    if verbose:
        print("\n📊 Creating validation and test splits...")
        print("━" * 50)
    
    # Load dev set
    if not DEV_POSTGRES_JSONL.exists():
        return {"error": f"Dev file not found: {DEV_POSTGRES_JSONL}"}
    
    dev_examples = load_jsonl(DEV_POSTGRES_JSONL)
    
    if verbose:
        print(f"Loaded dev set: {len(dev_examples)} examples")
    
    # Analyze complexity distribution
    if verbose:
        print("\nAnalyzing complexity distribution...")
        complexity_counts = defaultdict(int)
        for ex in dev_examples:
            sql = ex["messages"][2]["content"]
            level, _ = analyze_query_complexity(sql)
            complexity_counts[level] += 1
        
        total = len(dev_examples)
        for level in sorted(complexity_counts.keys()):
            label = get_complexity_label(level)
            count = complexity_counts[level]
            pct = count / total * 100
            print(f"  {label.capitalize()}: {count} ({pct:.1f}%)")
    
    # Perform stratified split
    if verbose:
        print("\nPerforming stratified split (50/50)...")
    
    val_examples, test_examples = stratified_split(dev_examples, split_ratio=0.5)
    
    if verbose:
        print(f"✅ Validation set: {len(val_examples)} examples")
        print(f"✅ Test set: {len(test_examples)} examples")
    
    # Save splits
    save_jsonl(val_examples, VAL_POSTGRES_JSONL)
    save_jsonl(test_examples, TEST_POSTGRES_JSONL)
    
    # Calculate statistics for all splits
    train_examples = load_jsonl(TRAIN_POSTGRES_JSONL) if TRAIN_POSTGRES_JSONL.exists() else []
    
    split_info = {
        "created_at": datetime.utcnow().isoformat(),
        "random_seed": RANDOM_SEED,
        "train": {
            "file": str(TRAIN_POSTGRES_JSONL.relative_to(PROJECT_DIR)),
            **get_split_stats(train_examples),
        },
        "validation": {
            "file": str(VAL_POSTGRES_JSONL.relative_to(PROJECT_DIR)),
            **get_split_stats(val_examples),
        },
        "test": {
            "file": str(TEST_POSTGRES_JSONL.relative_to(PROJECT_DIR)),
            **get_split_stats(test_examples),
        },
    }
    
    # Save split info
    with open(SPLIT_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    
    return split_info


def verify_splits(verbose: bool = True) -> bool:
    """
    Verify that splits have no overlap and similar distributions.
    
    Args:
        verbose: Whether to print details
        
    Returns:
        True if valid, False otherwise
    """
    if verbose:
        print("\n🔍 Verifying splits...")
    
    errors = []
    
    # Load all splits
    train_examples = load_jsonl(TRAIN_POSTGRES_JSONL) if TRAIN_POSTGRES_JSONL.exists() else []
    val_examples = load_jsonl(VAL_POSTGRES_JSONL) if VAL_POSTGRES_JSONL.exists() else []
    test_examples = load_jsonl(TEST_POSTGRES_JSONL) if TEST_POSTGRES_JSONL.exists() else []
    
    # Extract unique identifiers (question + SQL) for overlap check
    def get_example_ids(examples):
        ids = set()
        for ex in examples:
            # Use question + SQL as unique identifier
            user_content = ex["messages"][1]["content"]
            sql = ex["messages"][2]["content"]
            # Extract question from user content
            q_start = user_content.find("Question:")
            q_end = user_content.find("\n\nGenerate the SQL query:")
            question = user_content[q_start:q_end] if q_start != -1 and q_end != -1 else ""
            ids.add((question, sql))
        return ids
    
    train_ids = get_example_ids(train_examples)
    val_ids = get_example_ids(val_examples)
    test_ids = get_example_ids(test_examples)
    
    # Check overlaps
    train_val_overlap = len(train_ids & val_ids)
    train_test_overlap = len(train_ids & test_ids)
    val_test_overlap = len(val_ids & test_ids)
    
    # Val-Test overlap is critical (we control this split)
    if val_test_overlap > 0:
        errors.append(f"Val-Test overlap: {val_test_overlap} queries")
    
    # Train-Val/Test overlaps are warnings (Spider dataset characteristic)
    warnings = []
    if train_val_overlap > 0:
        warnings.append(f"Train-Val overlap: {train_val_overlap} queries (Spider dataset duplicate)")
    if train_test_overlap > 0:
        warnings.append(f"Train-Test overlap: {train_test_overlap} queries (Spider dataset duplicate)")
    
    if verbose:
        if val_test_overlap == 0:
            print("✅ No overlap between validation and test splits")
        if warnings:
            for warning in warnings:
                print(f"⚠️  {warning}")
        else:
            for error in errors:
                print(f"❌ {error}")
    
    # Check distribution similarity between val and test
    val_stats = get_split_stats(val_examples)
    test_stats = get_split_stats(test_examples)
    
    val_dist = val_stats["complexity_distribution"]
    test_dist = test_stats["complexity_distribution"]
    
    # Calculate distribution difference
    all_levels = set(val_dist.keys()) | set(test_dist.keys())
    max_diff = 0
    for level in all_levels:
        val_pct = val_dist.get(level, 0) / val_stats["count"] * 100 if val_stats["count"] > 0 else 0
        test_pct = test_dist.get(level, 0) / test_stats["count"] * 100 if test_stats["count"] > 0 else 0
        diff = abs(val_pct - test_pct)
        max_diff = max(max_diff, diff)
    
    if max_diff > 5:  # More than 5% difference
        errors.append(f"Distribution mismatch: {max_diff:.1f}% max difference")
    
    if verbose:
        if max_diff <= 5:
            print(f"✅ Distributions match (within {max_diff:.1f}%)")
        else:
            print(f"⚠️  Distribution difference: {max_diff:.1f}%")
    
    return len(errors) == 0


def main() -> int:
    """Main entry point for split creation."""
    print("📊 Creating final data splits...")
    
    # Create splits
    split_info = create_final_splits(verbose=True)
    
    if "error" in split_info:
        print(f"❌ Error: {split_info['error']}")
        return 1
    
    # Verify splits
    is_valid = verify_splits(verbose=True)
    
    # Print summary
    print("\n" + "━" * 50)
    print("✅ Split Creation Complete!")
    print("━" * 50)
    print(f"Training: {split_info['train']['count']:,} examples")
    print(f"Validation: {split_info['validation']['count']:,} examples")
    print(f"Test: {split_info['test']['count']:,} examples")
    print("━" * 50)
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
