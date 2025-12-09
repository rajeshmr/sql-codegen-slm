#!/usr/bin/env python3
"""
Tests for the Spider to Mistral format converter module.
"""

import json
import pytest
from pathlib import Path

from data.format_converter import (
    clean_sql_query,
    load_schema_content,
    format_mistral_example,
    validate_conversion,
    analyze_dataset,
    TRAIN_MISTRAL_JSONL,
    DEV_MISTRAL_JSONL,
    SCHEMA_INDEX_JSON,
    SCHEMAS_DIR,
    PROJECT_DIR,
    SYSTEM_PROMPT,
)


class TestCleanSqlQuery:
    """Tests for clean_sql_query function."""
    
    def test_removes_extra_whitespace(self):
        """Test that extra whitespace is removed."""
        sql = "SELECT   *   FROM   users   WHERE   id = 1"
        result = clean_sql_query(sql)
        assert "   " not in result
        assert result == "SELECT * FROM users WHERE id = 1;"
    
    def test_removes_newlines(self):
        """Test that newlines are normalized."""
        sql = """SELECT *
        FROM users
        WHERE id = 1"""
        result = clean_sql_query(sql)
        assert "\n" not in result
        assert result == "SELECT * FROM users WHERE id = 1;"
    
    def test_adds_semicolon(self):
        """Test that semicolon is added if missing."""
        sql = "SELECT * FROM users"
        result = clean_sql_query(sql)
        assert result.endswith(";")
    
    def test_keeps_existing_semicolon(self):
        """Test that existing semicolon is kept (no duplicate)."""
        sql = "SELECT * FROM users;"
        result = clean_sql_query(sql)
        assert result.count(";") == 1
    
    def test_handles_empty_string(self):
        """Test handling of empty string."""
        result = clean_sql_query("")
        assert result == ""
    
    def test_handles_tabs(self):
        """Test that tabs are normalized."""
        sql = "SELECT\t*\tFROM\tusers"
        result = clean_sql_query(sql)
        assert "\t" not in result


class TestLoadSchemaContent:
    """Tests for load_schema_content function."""
    
    @pytest.fixture
    def schema_index(self):
        """Load schema index for tests."""
        if not SCHEMA_INDEX_JSON.exists():
            pytest.skip("Schema index not found")
        with open(SCHEMA_INDEX_JSON, "r") as f:
            return json.load(f)
    
    def test_loads_academic_schema(self, schema_index):
        """Test loading academic database schema."""
        content = load_schema_content("academic", schema_index)
        
        assert content is not None
        assert "CREATE TABLE" in content
        assert "author" in content.lower()
    
    def test_returns_none_for_missing_db(self, schema_index):
        """Test that None is returned for non-existent database."""
        content = load_schema_content("nonexistent_database_xyz", schema_index)
        assert content is None
    
    def test_schema_has_pragma(self, schema_index):
        """Test that schema includes PRAGMA statement."""
        content = load_schema_content("academic", schema_index)
        
        assert content is not None
        assert "PRAGMA" in content.upper()


class TestFormatMistralExample:
    """Tests for format_mistral_example function."""
    
    def test_produces_correct_structure(self):
        """Test that output has correct Mistral format structure."""
        result = format_mistral_example(
            question="How many users are there?",
            sql_query="SELECT count(*) FROM users",
            schema_content="CREATE TABLE users (id int);",
            db_id="test_db"
        )
        
        assert "messages" in result
        assert len(result["messages"]) == 3
    
    def test_has_correct_roles(self):
        """Test that messages have correct roles."""
        result = format_mistral_example(
            question="Test question",
            sql_query="SELECT 1",
            schema_content="CREATE TABLE t (id int);",
            db_id="test"
        )
        
        roles = [msg["role"] for msg in result["messages"]]
        assert roles == ["system", "user", "assistant"]
    
    def test_system_message_content(self):
        """Test system message contains expected content."""
        result = format_mistral_example(
            question="Test",
            sql_query="SELECT 1",
            schema_content="CREATE TABLE t (id int);",
            db_id="test"
        )
        
        system_content = result["messages"][0]["content"]
        assert "PostgreSQL" in system_content
        assert "query generator" in system_content.lower()
    
    def test_user_message_contains_schema(self):
        """Test user message contains schema."""
        schema = "CREATE TABLE users (id int, name text);"
        result = format_mistral_example(
            question="How many users?",
            sql_query="SELECT count(*) FROM users",
            schema_content=schema,
            db_id="mydb"
        )
        
        user_content = result["messages"][1]["content"]
        assert "Schema:" in user_content
        assert schema in user_content
    
    def test_user_message_contains_database_name(self):
        """Test user message contains database name."""
        result = format_mistral_example(
            question="Test",
            sql_query="SELECT 1",
            schema_content="CREATE TABLE t (id int);",
            db_id="my_database"
        )
        
        user_content = result["messages"][1]["content"]
        assert "Database: my_database" in user_content
    
    def test_user_message_contains_question(self):
        """Test user message contains the question."""
        question = "How many active users are there?"
        result = format_mistral_example(
            question=question,
            sql_query="SELECT count(*) FROM users WHERE active = 1",
            schema_content="CREATE TABLE users (id int, active int);",
            db_id="test"
        )
        
        user_content = result["messages"][1]["content"]
        assert f"Question: {question}" in user_content
    
    def test_assistant_message_contains_sql(self):
        """Test assistant message contains cleaned SQL."""
        result = format_mistral_example(
            question="Test",
            sql_query="SELECT * FROM users",
            schema_content="CREATE TABLE users (id int);",
            db_id="test"
        )
        
        assistant_content = result["messages"][2]["content"]
        assert "SELECT * FROM users" in assistant_content
        assert assistant_content.endswith(";")
    
    def test_sql_is_cleaned(self):
        """Test that SQL in assistant message is cleaned."""
        result = format_mistral_example(
            question="Test",
            sql_query="SELECT   *   FROM   users",  # Extra spaces
            schema_content="CREATE TABLE users (id int);",
            db_id="test"
        )
        
        assistant_content = result["messages"][2]["content"]
        assert "   " not in assistant_content  # No extra spaces


class TestJsonlValidation:
    """Tests for JSONL file validation."""
    
    def test_train_jsonl_exists(self):
        """Test that training JSONL file exists."""
        if not TRAIN_MISTRAL_JSONL.exists():
            pytest.skip("Training JSONL not created yet")
        
        assert TRAIN_MISTRAL_JSONL.exists()
    
    def test_dev_jsonl_exists(self):
        """Test that dev JSONL file exists."""
        if not DEV_MISTRAL_JSONL.exists():
            pytest.skip("Dev JSONL not created yet")
        
        assert DEV_MISTRAL_JSONL.exists()
    
    def test_each_line_is_valid_json(self):
        """Test that each line in JSONL is valid JSON."""
        if not TRAIN_MISTRAL_JSONL.exists():
            pytest.skip("Training JSONL not created yet")
        
        with open(TRAIN_MISTRAL_JSONL, "r") as f:
            for i, line in enumerate(f):
                if i >= 100:  # Check first 100 lines
                    break
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Line {i+1} is not valid JSON: {e}")
    
    def test_jsonl_has_expected_structure(self):
        """Test that JSONL entries have expected structure."""
        if not TRAIN_MISTRAL_JSONL.exists():
            pytest.skip("Training JSONL not created yet")
        
        with open(TRAIN_MISTRAL_JSONL, "r") as f:
            example = json.loads(f.readline())
        
        assert "messages" in example
        assert len(example["messages"]) == 3
        
        for msg in example["messages"]:
            assert "role" in msg
            assert "content" in msg
    
    def test_reasonable_example_count(self):
        """Test that JSONL has reasonable number of examples."""
        if not TRAIN_MISTRAL_JSONL.exists():
            pytest.skip("Training JSONL not created yet")
        
        with open(TRAIN_MISTRAL_JSONL, "r") as f:
            count = sum(1 for _ in f)
        
        # Should have at least 6000 examples (some may be skipped due to missing schemas)
        assert count >= 6000, f"Only {count} examples, expected at least 6000"
        # Should not exceed original count
        assert count <= 8000, f"Got {count} examples, expected at most 8000"


class TestSchemaEmbedding:
    """Tests for schema content embedding in examples."""
    
    def test_schema_is_complete(self):
        """Test that schema in user message is complete (not truncated)."""
        if not TRAIN_MISTRAL_JSONL.exists():
            pytest.skip("Training JSONL not created yet")
        
        with open(TRAIN_MISTRAL_JSONL, "r") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                
                example = json.loads(line)
                user_content = example["messages"][1]["content"]
                
                # Should have at least one CREATE TABLE
                assert "CREATE TABLE" in user_content
                
                # Should have closing parenthesis and semicolon (complete statement)
                assert ");" in user_content or ") ;" in user_content
    
    def test_schema_matches_database(self):
        """Test that schema content matches the database mentioned."""
        if not TRAIN_MISTRAL_JSONL.exists():
            pytest.skip("Training JSONL not created yet")
        if not SCHEMA_INDEX_JSON.exists():
            pytest.skip("Schema index not found")
        
        with open(SCHEMA_INDEX_JSON, "r") as f:
            schema_index = json.load(f)
        
        with open(TRAIN_MISTRAL_JSONL, "r") as f:
            example = json.loads(f.readline())
        
        user_content = example["messages"][1]["content"]
        
        # Extract database name
        db_line = user_content.split("\n")[0]
        db_id = db_line.replace("Database:", "").strip()
        
        # Get expected tables from schema index
        if db_id in schema_index:
            expected_tables = schema_index[db_id]["tables"]
            
            # At least some tables should be mentioned in schema
            tables_found = sum(1 for t in expected_tables if t.lower() in user_content.lower())
            assert tables_found > 0, f"No tables from {db_id} found in schema"


class TestValidationFunction:
    """Tests for the validate_conversion function."""
    
    def test_validation_runs(self):
        """Test that validation function runs without error."""
        if not TRAIN_MISTRAL_JSONL.exists():
            pytest.skip("Training JSONL not created yet")
        
        # Should not raise exception
        result = validate_conversion(verbose=False)
        assert isinstance(result, bool)


class TestAnalyzeDataset:
    """Tests for dataset analysis function."""
    
    def test_analyze_returns_statistics(self):
        """Test that analyze_dataset returns statistics."""
        if not TRAIN_MISTRAL_JSONL.exists():
            pytest.skip("Training JSONL not created yet")
        
        stats = analyze_dataset(verbose=False)
        
        assert "total_examples" in stats
        assert "with_join" in stats
        assert "with_group_by" in stats
        assert stats["total_examples"] > 0
    
    def test_statistics_are_reasonable(self):
        """Test that statistics are within reasonable ranges."""
        if not TRAIN_MISTRAL_JSONL.exists():
            pytest.skip("Training JSONL not created yet")
        
        stats = analyze_dataset(verbose=False)
        
        total = stats["total_examples"]
        
        # Simple SELECT should be less than 70% (Spider has complex queries)
        # Note: Actual percentage depends on which schemas are available
        assert stats["simple_select"] < total * 0.70
        
        # Should have some JOINs (at least 20%)
        assert stats["with_join"] > total * 0.20
        
        # Should have some GROUP BY queries (at least 15%)
        assert stats["with_group_by"] > total * 0.15
