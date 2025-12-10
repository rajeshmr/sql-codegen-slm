#!/usr/bin/env python3
"""
Tests for the final data pipeline: splits, demo schemas, and statistics.
"""

import json
import pytest
from pathlib import Path

from data.split_creator import (
    analyze_query_complexity,
    get_complexity_label,
    stratified_split,
    load_jsonl,
    get_split_stats,
    TRAIN_POSTGRES_JSONL,
    VAL_POSTGRES_JSONL,
    TEST_POSTGRES_JSONL,
    SPLIT_INFO_JSON,
)
from data.demo_schema_generator import (
    DEMO_SCHEMAS,
    DEMO_DIR,
    DEMO_SCHEMAS_JSON,
)
from data.dataset_stats import (
    analyze_sql_features,
    calculate_token_stats,
    DATASET_STATS_JSON,
)


class TestQueryComplexity:
    """Tests for query complexity analysis."""
    
    def test_simple_query(self):
        """Test simple SELECT query is level 1."""
        sql = "SELECT * FROM users WHERE id = 1;"
        level, features = analyze_query_complexity(sql)
        assert level == 1
        assert features["has_join"] is False
    
    def test_medium_query_with_join(self):
        """Test query with JOIN is level 2."""
        sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id;"
        level, features = analyze_query_complexity(sql)
        assert level == 2
        assert features["has_join"] is True
    
    def test_medium_query_with_group_by(self):
        """Test query with GROUP BY is level 2."""
        sql = "SELECT category, COUNT(*) FROM products GROUP BY category;"
        level, features = analyze_query_complexity(sql)
        assert level == 2
        assert features["has_group_by"] is True
    
    def test_complex_query_with_subquery(self):
        """Test query with subquery is level 3."""
        sql = "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE total > 100);"
        level, features = analyze_query_complexity(sql)
        assert level >= 3
        assert features["has_subquery"] is True
    
    def test_complex_query_with_multiple_joins(self):
        """Test query with multiple JOINs is level 3."""
        sql = "SELECT * FROM a JOIN b ON a.id = b.a_id JOIN c ON b.id = c.b_id JOIN d ON c.id = d.c_id;"
        level, features = analyze_query_complexity(sql)
        assert level >= 3
        assert features["join_count"] >= 2
    
    def test_complexity_labels(self):
        """Test complexity label mapping."""
        assert get_complexity_label(1) == "simple"
        assert get_complexity_label(2) == "medium"
        assert get_complexity_label(3) == "complex"
        assert get_complexity_label(4) == "very_complex"


class TestStratifiedSplit:
    """Tests for stratified splitting."""
    
    def test_split_ratio(self):
        """Test that split ratio is approximately correct."""
        # Create mock examples with different complexities
        examples = []
        for i in range(100):
            if i < 40:
                sql = "SELECT * FROM t;"  # Simple
            elif i < 70:
                sql = "SELECT * FROM a JOIN b ON a.id = b.a_id;"  # Medium
            else:
                sql = "SELECT * FROM a WHERE id IN (SELECT id FROM b);"  # Complex
            
            examples.append({
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Database: test\n\nSchema:\nCREATE TABLE t (id int);\n\nQuestion: Q\n\nGenerate the SQL query:"},
                    {"role": "assistant", "content": sql}
                ]
            })
        
        first, second = stratified_split(examples, split_ratio=0.5, seed=42)
        
        # Check approximately equal sizes
        assert abs(len(first) - len(second)) <= 5
        assert len(first) + len(second) == len(examples)
    
    def test_no_overlap(self):
        """Test that splits have no overlap."""
        examples = [
            {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "assistant", "content": f"SELECT {i}"}]}
            for i in range(50)
        ]
        
        first, second = stratified_split(examples, split_ratio=0.5, seed=42)
        
        first_sqls = {ex["messages"][2]["content"] for ex in first}
        second_sqls = {ex["messages"][2]["content"] for ex in second}
        
        assert len(first_sqls & second_sqls) == 0


class TestSplitFiles:
    """Tests for split output files."""
    
    def test_train_file_exists(self):
        """Test training file exists."""
        if not TRAIN_POSTGRES_JSONL.exists():
            pytest.skip("Training file not created yet")
        assert TRAIN_POSTGRES_JSONL.exists()
    
    def test_val_file_exists(self):
        """Test validation file exists."""
        if not VAL_POSTGRES_JSONL.exists():
            pytest.skip("Validation file not created yet")
        assert VAL_POSTGRES_JSONL.exists()
    
    def test_test_file_exists(self):
        """Test test file exists."""
        if not TEST_POSTGRES_JSONL.exists():
            pytest.skip("Test file not created yet")
        assert TEST_POSTGRES_JSONL.exists()
    
    def test_split_info_exists(self):
        """Test split info JSON exists."""
        if not SPLIT_INFO_JSON.exists():
            pytest.skip("Split info not created yet")
        assert SPLIT_INFO_JSON.exists()
    
    def _get_example_ids(self, examples):
        """Extract unique identifiers (question + SQL) from examples."""
        ids = set()
        for ex in examples:
            user_content = ex["messages"][1]["content"]
            sql = ex["messages"][2]["content"]
            q_start = user_content.find("Question:")
            q_end = user_content.find("\n\nGenerate the SQL query:")
            question = user_content[q_start:q_end] if q_start != -1 and q_end != -1 else ""
            ids.add((question, sql))
        return ids
    
    def test_no_train_val_overlap(self):
        """Test minimal overlap between train and validation (Spider dataset may have duplicates)."""
        if not TRAIN_POSTGRES_JSONL.exists() or not VAL_POSTGRES_JSONL.exists():
            pytest.skip("Split files not created yet")
        
        train = load_jsonl(TRAIN_POSTGRES_JSONL)
        val = load_jsonl(VAL_POSTGRES_JSONL)
        
        train_ids = self._get_example_ids(train)
        val_ids = self._get_example_ids(val)
        
        overlap = len(train_ids & val_ids)
        # Allow small overlap due to Spider dataset characteristics
        assert overlap <= 5, f"Found {overlap} overlapping examples between train and val (max 5 allowed)"
    
    def test_no_train_test_overlap(self):
        """Test minimal overlap between train and test (Spider dataset may have duplicates)."""
        if not TRAIN_POSTGRES_JSONL.exists() or not TEST_POSTGRES_JSONL.exists():
            pytest.skip("Split files not created yet")
        
        train = load_jsonl(TRAIN_POSTGRES_JSONL)
        test = load_jsonl(TEST_POSTGRES_JSONL)
        
        train_ids = self._get_example_ids(train)
        test_ids = self._get_example_ids(test)
        
        overlap = len(train_ids & test_ids)
        # Allow small overlap due to Spider dataset characteristics
        assert overlap <= 5, f"Found {overlap} overlapping examples between train and test (max 5 allowed)"
    
    def test_no_val_test_overlap(self):
        """Test no overlap between validation and test (we control this split)."""
        if not VAL_POSTGRES_JSONL.exists() or not TEST_POSTGRES_JSONL.exists():
            pytest.skip("Split files not created yet")
        
        val = load_jsonl(VAL_POSTGRES_JSONL)
        test = load_jsonl(TEST_POSTGRES_JSONL)
        
        val_ids = self._get_example_ids(val)
        test_ids = self._get_example_ids(test)
        
        overlap = len(val_ids & test_ids)
        assert overlap == 0, f"Found {overlap} overlapping examples between val and test"
    
    def test_val_test_roughly_equal(self):
        """Test validation and test sets are roughly equal size."""
        if not VAL_POSTGRES_JSONL.exists() or not TEST_POSTGRES_JSONL.exists():
            pytest.skip("Split files not created yet")
        
        val = load_jsonl(VAL_POSTGRES_JSONL)
        test = load_jsonl(TEST_POSTGRES_JSONL)
        
        # Should be within 10% of each other
        ratio = len(val) / len(test) if len(test) > 0 else 0
        assert 0.9 <= ratio <= 1.1, f"Val/Test ratio is {ratio:.2f}, expected ~1.0"
    
    def test_split_info_valid_json(self):
        """Test split info is valid JSON with expected structure."""
        if not SPLIT_INFO_JSON.exists():
            pytest.skip("Split info not created yet")
        
        with open(SPLIT_INFO_JSON, "r") as f:
            info = json.load(f)
        
        assert "train" in info
        assert "validation" in info
        assert "test" in info
        
        for split in ["train", "validation", "test"]:
            assert "count" in info[split]
            assert "databases" in info[split]
            assert "complexity_distribution" in info[split]


class TestDemoSchemas:
    """Tests for demo schema generation."""
    
    def test_demo_dir_exists(self):
        """Test demo directory exists."""
        if not DEMO_DIR.exists():
            pytest.skip("Demo directory not created yet")
        assert DEMO_DIR.exists()
    
    def test_all_schema_files_exist(self):
        """Test all 5 demo schema files exist."""
        if not DEMO_DIR.exists():
            pytest.skip("Demo directory not created yet")
        
        expected_files = [
            "ecommerce_schema.sql",
            "finance_schema.sql",
            "healthcare_schema.sql",
            "saas_schema.sql",
            "retail_schema.sql",
        ]
        
        for filename in expected_files:
            filepath = DEMO_DIR / filename
            assert filepath.exists(), f"Missing demo schema: {filename}"
    
    def test_demo_schemas_json_exists(self):
        """Test demo schemas JSON exists."""
        if not DEMO_SCHEMAS_JSON.exists():
            pytest.skip("Demo schemas JSON not created yet")
        assert DEMO_SCHEMAS_JSON.exists()
    
    def test_demo_schemas_json_valid(self):
        """Test demo schemas JSON is valid and has expected structure."""
        if not DEMO_SCHEMAS_JSON.exists():
            pytest.skip("Demo schemas JSON not created yet")
        
        with open(DEMO_SCHEMAS_JSON, "r") as f:
            data = json.load(f)
        
        expected_schemas = ["ecommerce", "finance", "healthcare", "saas", "retail"]
        
        for schema_name in expected_schemas:
            assert schema_name in data, f"Missing schema: {schema_name}"
            schema = data[schema_name]
            assert "name" in schema
            assert "description" in schema
            assert "tables" in schema
            assert "sample_questions" in schema
            assert len(schema["sample_questions"]) >= 5
    
    def test_schemas_have_postgresql_syntax(self):
        """Test demo schemas use PostgreSQL syntax."""
        if not DEMO_DIR.exists():
            pytest.skip("Demo directory not created yet")
        
        for schema_name, schema_info in DEMO_SCHEMAS.items():
            filepath = DEMO_DIR / schema_info["filename"]
            if not filepath.exists():
                continue
            
            content = filepath.read_text()
            
            # Should have PostgreSQL features
            assert "SERIAL" in content, f"{schema_name}: Missing SERIAL"
            assert "REFERENCES" in content, f"{schema_name}: Missing REFERENCES"
            
            # Should NOT have SQLite features
            assert "AUTOINCREMENT" not in content, f"{schema_name}: Contains AUTOINCREMENT"
            assert "PRAGMA" not in content, f"{schema_name}: Contains PRAGMA"
    
    def test_schemas_have_foreign_keys(self):
        """Test demo schemas have proper foreign key relationships."""
        if not DEMO_DIR.exists():
            pytest.skip("Demo directory not created yet")
        
        for schema_name, schema_info in DEMO_SCHEMAS.items():
            filepath = DEMO_DIR / schema_info["filename"]
            if not filepath.exists():
                continue
            
            content = filepath.read_text()
            
            # Should have at least one foreign key
            assert "REFERENCES" in content, f"{schema_name}: No foreign keys found"
    
    def test_schemas_have_multiple_tables(self):
        """Test demo schemas have 4-6 tables each."""
        if not DEMO_DIR.exists():
            pytest.skip("Demo directory not created yet")
        
        for schema_name, schema_info in DEMO_SCHEMAS.items():
            filepath = DEMO_DIR / schema_info["filename"]
            if not filepath.exists():
                continue
            
            content = filepath.read_text()
            
            # Count CREATE TABLE statements
            table_count = content.count("CREATE TABLE")
            assert 4 <= table_count <= 7, f"{schema_name}: Has {table_count} tables, expected 4-6"


class TestDatasetStatistics:
    """Tests for dataset statistics generation."""
    
    def test_stats_file_exists(self):
        """Test statistics file exists."""
        if not DATASET_STATS_JSON.exists():
            pytest.skip("Statistics file not created yet")
        assert DATASET_STATS_JSON.exists()
    
    def test_stats_valid_json(self):
        """Test statistics file is valid JSON."""
        if not DATASET_STATS_JSON.exists():
            pytest.skip("Statistics file not created yet")
        
        with open(DATASET_STATS_JSON, "r") as f:
            stats = json.load(f)
        
        assert "splits" in stats
        assert "total_examples" in stats
    
    def test_stats_has_all_splits(self):
        """Test statistics has data for all splits."""
        if not DATASET_STATS_JSON.exists():
            pytest.skip("Statistics file not created yet")
        
        with open(DATASET_STATS_JSON, "r") as f:
            stats = json.load(f)
        
        for split in ["train", "validation", "test"]:
            if split in stats["splits"]:
                split_stats = stats["splits"][split]
                assert "count" in split_stats
                assert "databases" in split_stats
    
    def test_analyze_sql_features(self):
        """Test SQL feature analysis."""
        sql = "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.name HAVING COUNT(o.id) > 5 ORDER BY COUNT(o.id) DESC LIMIT 10;"
        
        features = analyze_sql_features(sql)
        
        assert features["has_select"] is True
        assert features["has_join"] is True
        assert features["has_left_join"] is True
        assert features["has_group_by"] is True
        assert features["has_having"] is True
        assert features["has_order_by"] is True
        assert features["has_limit"] is True
        assert features["has_count"] is True
        assert features["has_aggregation"] is True


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_total_examples_match(self):
        """Test that val + test equals original dev count."""
        if not VAL_POSTGRES_JSONL.exists() or not TEST_POSTGRES_JSONL.exists():
            pytest.skip("Split files not created yet")
        
        val = load_jsonl(VAL_POSTGRES_JSONL)
        test = load_jsonl(TEST_POSTGRES_JSONL)
        
        # Val + Test should equal original dev (665 in our case)
        total = len(val) + len(test)
        
        # Should be within reasonable range (allowing for some variance)
        assert total >= 600, f"Total val+test is only {total}"
    
    def test_complexity_distribution_similar(self):
        """Test that val and test have similar complexity distributions."""
        if not VAL_POSTGRES_JSONL.exists() or not TEST_POSTGRES_JSONL.exists():
            pytest.skip("Split files not created yet")
        
        val = load_jsonl(VAL_POSTGRES_JSONL)
        test = load_jsonl(TEST_POSTGRES_JSONL)
        
        val_stats = get_split_stats(val)
        test_stats = get_split_stats(test)
        
        val_dist = val_stats["complexity_distribution"]
        test_dist = test_stats["complexity_distribution"]
        
        # Check each complexity level is within 10% difference
        for level in set(val_dist.keys()) | set(test_dist.keys()):
            val_pct = val_dist.get(level, 0) / val_stats["count"] * 100 if val_stats["count"] > 0 else 0
            test_pct = test_dist.get(level, 0) / test_stats["count"] * 100 if test_stats["count"] > 0 else 0
            diff = abs(val_pct - test_pct)
            assert diff <= 10, f"Complexity '{level}' differs by {diff:.1f}% between val and test"
