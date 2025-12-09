#!/usr/bin/env python3
"""
Schema Parser for Spider Dataset.

This module parses and indexes the pre-existing schema.sql files from the Spider dataset.
It extracts metadata about tables, columns, and foreign key relationships.
"""

import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
SPIDER_DB_DIR = PROJECT_DIR / "data" / "raw" / "spider" / "database"
PROCESSED_SCHEMAS_DIR = PROJECT_DIR / "data" / "processed" / "schemas"
TABLES_JSON_PATH = PROJECT_DIR / "data" / "raw" / "spider" / "tables.json"


def parse_schema_file(schema_path: Path) -> dict[str, Any]:
    """
    Parse a schema.sql file and extract metadata.
    
    Args:
        schema_path: Path to the schema.sql file
        
    Returns:
        Dictionary with parsed metadata including tables, columns, and foreign keys
    """
    if not schema_path.exists():
        return {"error": f"File not found: {schema_path}"}
    
    try:
        content = schema_path.read_text(encoding="utf-8")
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}
    
    # Extract table names - handle various quote styles
    # Matches: CREATE TABLE "name", CREATE TABLE 'name', CREATE TABLE `name`, CREATE TABLE name
    table_pattern = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["\'\`]?(\w+)["\'\`]?\s*\(',
        re.IGNORECASE
    )
    tables = table_pattern.findall(content)
    
    # Extract columns for each table
    table_columns = {}
    # Pattern to match CREATE TABLE ... ( ... ) with content between parentheses
    table_block_pattern = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["\'\`]?(\w+)["\'\`]?\s*\((.*?)\);',
        re.IGNORECASE | re.DOTALL
    )
    
    for match in table_block_pattern.finditer(content):
        table_name = match.group(1)
        table_body = match.group(2)
        
        # Extract column definitions (lines that don't start with PRIMARY KEY, FOREIGN KEY, etc.)
        columns = []
        for line in table_body.split('\n'):
            line = line.strip().rstrip(',')
            if not line:
                continue
            # Skip constraint definitions
            if re.match(r'^\s*(primary\s+key|foreign\s+key|unique|check|constraint)', line, re.IGNORECASE):
                continue
            # Extract column name (first word, possibly quoted)
            col_match = re.match(r'["\'\`]?(\w+)["\'\`]?\s+', line)
            if col_match:
                columns.append(col_match.group(1))
        
        table_columns[table_name] = columns
    
    # Detect foreign keys
    has_foreign_keys = bool(re.search(r'foreign\s+key', content, re.IGNORECASE))
    
    # Extract foreign key relationships
    foreign_keys = []
    fk_pattern = re.compile(
        r'foreign\s+key\s*\(["\'\`]?(\w+)["\'\`]?\)\s*references\s+["\'\`]?(\w+)["\'\`]?\s*\(["\'\`]?(\w+)["\'\`]?\)',
        re.IGNORECASE
    )
    for match in fk_pattern.finditer(content):
        foreign_keys.append({
            "column": match.group(1),
            "references_table": match.group(2),
            "references_column": match.group(3)
        })
    
    # Detect primary keys
    has_primary_keys = bool(re.search(r'primary\s+key', content, re.IGNORECASE))
    
    # Check for PRAGMA statement
    has_pragma = content.strip().upper().startswith("PRAGMA")
    
    return {
        "tables": tables,
        "num_tables": len(tables),
        "table_columns": table_columns,
        "has_foreign_keys": has_foreign_keys,
        "has_primary_keys": has_primary_keys,
        "foreign_keys": foreign_keys,
        "has_pragma": has_pragma,
        "content_length": len(content),
    }


def validate_schema_content(schema_content: str) -> tuple[bool, list[str]]:
    """
    Validate schema content for basic SQL syntax.
    
    Args:
        schema_content: Raw SQL schema content
        
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []
    
    if not schema_content or not schema_content.strip():
        return False, ["Schema content is empty"]
    
    # Check for at least one CREATE TABLE statement
    if not re.search(r'CREATE\s+TABLE', schema_content, re.IGNORECASE):
        return False, ["No CREATE TABLE statement found"]
    
    # Check for PRAGMA statement (optional but recommended)
    if not schema_content.strip().upper().startswith("PRAGMA"):
        warnings.append("Schema does not start with PRAGMA statement")
    
    # Check for balanced parentheses (basic syntax check)
    open_parens = schema_content.count('(')
    close_parens = schema_content.count(')')
    if open_parens != close_parens:
        warnings.append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")
    
    # Check for semicolons (statements should end with semicolons)
    if ';' not in schema_content:
        warnings.append("No semicolons found - statements may be incomplete")
    
    is_valid = len([w for w in warnings if "No CREATE TABLE" in w]) == 0
    return is_valid, warnings


def parse_all_schemas(verbose: bool = True) -> dict[str, Any]:
    """
    Parse all schema.sql files from Spider database directories.
    
    Args:
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with parsing statistics and results
    """
    # Ensure output directory exists
    PROCESSED_SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all database directories
    if not SPIDER_DB_DIR.exists():
        return {
            "error": f"Spider database directory not found: {SPIDER_DB_DIR}",
            "total_found": 0,
            "successfully_parsed": 0,
        }
    
    db_dirs = sorted([d for d in SPIDER_DB_DIR.iterdir() if d.is_dir()])
    
    # Statistics
    stats = {
        "total_found": len(db_dirs),
        "schema_files_found": 0,
        "successfully_parsed": 0,
        "missing_schema": 0,
        "parse_errors": 0,
        "databases_lt_5_tables": 0,
        "databases_5_to_10_tables": 0,
        "databases_gt_10_tables": 0,
        "databases_with_foreign_keys": 0,
    }
    
    # Master index
    schema_index = {}
    
    # Process each database directory
    iterator = tqdm(db_dirs, desc="Processing schemas", disable=not verbose)
    
    for db_dir in iterator:
        db_name = db_dir.name
        schema_path = db_dir / "schema.sql"
        
        if not schema_path.exists():
            stats["missing_schema"] += 1
            continue
        
        stats["schema_files_found"] += 1
        
        # Parse the schema
        parsed = parse_schema_file(schema_path)
        
        if "error" in parsed:
            stats["parse_errors"] += 1
            continue
        
        # Copy schema to processed directory
        dest_path = PROCESSED_SCHEMAS_DIR / f"{db_name}_schema.sql"
        try:
            shutil.copy2(schema_path, dest_path)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to copy {schema_path}: {e}")
        
        # Update statistics
        stats["successfully_parsed"] += 1
        
        num_tables = parsed["num_tables"]
        if num_tables < 5:
            stats["databases_lt_5_tables"] += 1
        elif num_tables <= 10:
            stats["databases_5_to_10_tables"] += 1
        else:
            stats["databases_gt_10_tables"] += 1
        
        if parsed["has_foreign_keys"]:
            stats["databases_with_foreign_keys"] += 1
        
        # Add to index
        schema_index[db_name] = {
            "schema_file": str(dest_path.relative_to(PROJECT_DIR)),
            "source_file": str(schema_path.relative_to(PROJECT_DIR)),
            "num_tables": parsed["num_tables"],
            "tables": parsed["tables"],
            "table_columns": parsed["table_columns"],
            "has_foreign_keys": parsed["has_foreign_keys"],
            "has_primary_keys": parsed["has_primary_keys"],
            "foreign_keys": parsed["foreign_keys"],
            "parsed_timestamp": datetime.utcnow().isoformat(),
        }
    
    # Save schema index
    index_path = PROCESSED_SCHEMAS_DIR / "schema_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(schema_index, f, indent=2)
    
    stats["index_path"] = str(index_path.relative_to(PROJECT_DIR))
    stats["schemas_dir"] = str(PROCESSED_SCHEMAS_DIR.relative_to(PROJECT_DIR))
    
    return stats, schema_index


def cross_reference_tables_json(schema_index: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """
    Cross-reference parsed schemas with tables.json from Spider dataset.
    
    Args:
        schema_index: Dictionary of parsed schema metadata
        verbose: Whether to print details
        
    Returns:
        Dictionary with validation results
    """
    if not TABLES_JSON_PATH.exists():
        return {"error": f"tables.json not found: {TABLES_JSON_PATH}"}
    
    with open(TABLES_JSON_PATH, "r", encoding="utf-8") as f:
        tables_json = json.load(f)
    
    # Build lookup from tables.json
    tables_json_dbs = {}
    for db_info in tables_json:
        db_id = db_info.get("db_id", "")
        table_names = [t.lower() for t in db_info.get("table_names_original", [])]
        tables_json_dbs[db_id] = table_names
    
    results = {
        "total_in_tables_json": len(tables_json_dbs),
        "total_in_schema_index": len(schema_index),
        "matched": 0,
        "in_tables_json_only": [],
        "in_schema_index_only": [],
        "table_mismatches": [],
    }
    
    # Check databases in tables.json
    for db_id, expected_tables in tables_json_dbs.items():
        if db_id in schema_index:
            results["matched"] += 1
            
            # Compare table names
            parsed_tables = [t.lower() for t in schema_index[db_id]["tables"]]
            expected_set = set(expected_tables)
            parsed_set = set(parsed_tables)
            
            if expected_set != parsed_set:
                missing = expected_set - parsed_set
                extra = parsed_set - expected_set
                if missing or extra:
                    results["table_mismatches"].append({
                        "db_id": db_id,
                        "missing_from_schema": list(missing),
                        "extra_in_schema": list(extra),
                    })
        else:
            results["in_tables_json_only"].append(db_id)
    
    # Check databases in schema_index but not in tables.json
    for db_id in schema_index:
        if db_id not in tables_json_dbs:
            results["in_schema_index_only"].append(db_id)
    
    if verbose:
        print(f"\n📊 Cross-Reference Validation:")
        print(f"   Databases in tables.json: {results['total_in_tables_json']}")
        print(f"   Databases in schema_index: {results['total_in_schema_index']}")
        print(f"   Matched: {results['matched']}")
        print(f"   In tables.json only: {len(results['in_tables_json_only'])}")
        print(f"   In schema_index only: {len(results['in_schema_index_only'])}")
        print(f"   Table name mismatches: {len(results['table_mismatches'])}")
        
        if results["table_mismatches"] and len(results["table_mismatches"]) <= 5:
            print("\n   Sample mismatches:")
            for mismatch in results["table_mismatches"][:3]:
                print(f"   - {mismatch['db_id']}: missing={mismatch['missing_from_schema']}, extra={mismatch['extra_in_schema']}")
    
    return results


def print_summary(stats: dict[str, Any]) -> None:
    """Print parsing summary statistics."""
    print("\n" + "━" * 50)
    print("✅ Parsing Complete!")
    print("━" * 50)
    print(f"Total directories found:     {stats['total_found']}")
    print(f"Schema files found:          {stats['schema_files_found']}")
    print(f"Successfully parsed:         {stats['successfully_parsed']}")
    print(f"Missing schema.sql:          {stats['missing_schema']}")
    print(f"Parse errors:                {stats['parse_errors']}")
    print(f"Schemas copied to:           {stats['schemas_dir']}")
    print(f"Index created:               {stats['index_path']}")
    print("━" * 50)
    print("\n📊 Schema Statistics:")
    print(f"   Databases with <5 tables:    {stats['databases_lt_5_tables']}")
    print(f"   Databases with 5-10 tables:  {stats['databases_5_to_10_tables']}")
    print(f"   Databases with >10 tables:   {stats['databases_gt_10_tables']}")
    print(f"   Databases with foreign keys: {stats['databases_with_foreign_keys']}")


def main() -> int:
    """Main entry point for schema parsing."""
    print("🔍 Parsing Spider schema files...")
    
    # Parse all schemas
    result = parse_all_schemas(verbose=True)
    
    if isinstance(result, tuple):
        stats, schema_index = result
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
        return 1
    
    # Print summary
    print_summary(stats)
    
    # Cross-reference with tables.json
    print("\n🔄 Running cross-reference validation...")
    validation = cross_reference_tables_json(schema_index, verbose=True)
    
    if "error" in validation:
        print(f"⚠️  Validation warning: {validation['error']}")
    else:
        match_rate = validation["matched"] / max(validation["total_in_tables_json"], 1) * 100
        print(f"\n✅ {validation['matched']}/{validation['total_in_tables_json']} databases match tables.json ({match_rate:.1f}%)")
        
        if validation["table_mismatches"]:
            print(f"⚠️  {len(validation['table_mismatches'])} minor discrepancies found")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
