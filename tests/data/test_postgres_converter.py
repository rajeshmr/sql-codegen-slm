#!/usr/bin/env python3
"""
Tests for the SQLite to PostgreSQL converter module.
"""

import json
import pytest
from pathlib import Path

from data.postgres_converter import (
    convert_sql_query,
    convert_schema,
    detect_conversion_patterns,
    validate_postgres_syntax,
    convert_example,
    TRAIN_POSTGRES_JSONL,
    DEV_POSTGRES_JSONL,
    TRAIN_MISTRAL_JSONL,
)


class TestConvertSqlQuery:
    """Tests for convert_sql_query function."""
    
    def test_datetime_now_conversion(self):
        """Test DATETIME('now') → CURRENT_TIMESTAMP."""
        sql = "SELECT * FROM users WHERE created_at > DATETIME('now')"
        result = convert_sql_query(sql)
        assert "CURRENT_TIMESTAMP" in result
        assert "DATETIME('now')" not in result
    
    def test_datetime_now_case_insensitive(self):
        """Test conversion is case-insensitive."""
        sql = "SELECT * FROM users WHERE created_at > datetime('now')"
        result = convert_sql_query(sql)
        assert "CURRENT_TIMESTAMP" in result
    
    def test_date_now_conversion(self):
        """Test DATE('now') → CURRENT_DATE."""
        sql = "SELECT * FROM users WHERE birth_date = DATE('now')"
        result = convert_sql_query(sql)
        assert "CURRENT_DATE" in result
        assert "DATE('now')" not in result
    
    def test_time_now_conversion(self):
        """Test TIME('now') → CURRENT_TIME."""
        sql = "SELECT * FROM logs WHERE log_time = TIME('now')"
        result = convert_sql_query(sql)
        assert "CURRENT_TIME" in result
        assert "TIME('now')" not in result
    
    def test_strftime_conversion(self):
        """Test strftime → TO_CHAR conversion."""
        sql = "SELECT strftime('%Y-%m-%d', created_at) FROM users"
        result = convert_sql_query(sql)
        assert "TO_CHAR" in result
        assert "YYYY-MM-DD" in result
        assert "strftime" not in result.lower()
    
    def test_substr_conversion(self):
        """Test SUBSTR → SUBSTRING conversion."""
        sql = "SELECT SUBSTR(name, 1, 3) FROM users"
        result = convert_sql_query(sql)
        assert "SUBSTRING" in result
        assert "FROM 1 FOR 3" in result
        assert "SUBSTR(" not in result  # Check for function call, not substring of SUBSTRING
    
    def test_concatenation_kept(self):
        """Test that || concatenation is kept (PostgreSQL supports it)."""
        sql = "SELECT first_name || ' ' || last_name FROM users"
        result = convert_sql_query(sql)
        assert "||" in result
    
    def test_no_conversion_needed(self):
        """Test query that needs no conversion."""
        sql = "SELECT * FROM users WHERE id = 1"
        result = convert_sql_query(sql)
        assert result == sql
    
    def test_multiple_conversions(self):
        """Test query with multiple patterns to convert."""
        sql = "SELECT SUBSTR(name, 1, 5), DATETIME('now') FROM users"
        result = convert_sql_query(sql)
        assert "SUBSTRING" in result
        assert "CURRENT_TIMESTAMP" in result
        assert "SUBSTR(" not in result  # Check for function call, not substring of SUBSTRING
        assert "DATETIME('now')" not in result
    
    def test_empty_query(self):
        """Test handling of empty query."""
        result = convert_sql_query("")
        assert result == ""
    
    def test_none_query(self):
        """Test handling of None query."""
        result = convert_sql_query(None)
        assert result is None


class TestConvertSchema:
    """Tests for convert_schema function."""
    
    def test_pragma_removal(self):
        """Test PRAGMA statement removal."""
        schema = "PRAGMA foreign_keys = ON;\nCREATE TABLE users (id int);"
        result = convert_schema(schema)
        assert "PRAGMA" not in result
        assert "CREATE TABLE" in result
    
    def test_autoincrement_conversion(self):
        """Test INTEGER PRIMARY KEY AUTOINCREMENT → SERIAL PRIMARY KEY."""
        schema = 'CREATE TABLE users ("id" INTEGER PRIMARY KEY AUTOINCREMENT, "name" TEXT);'
        result = convert_schema(schema)
        assert "SERIAL PRIMARY KEY" in result
        assert "AUTOINCREMENT" not in result
    
    def test_blob_to_bytea(self):
        """Test BLOB → BYTEA conversion."""
        schema = 'CREATE TABLE files ("id" int, "data" BLOB);'
        result = convert_schema(schema)
        assert "BYTEA" in result
        assert "BLOB" not in result
    
    def test_real_to_double_precision(self):
        """Test REAL → DOUBLE PRECISION conversion."""
        schema = 'CREATE TABLE measurements ("id" int, "value" REAL);'
        result = convert_schema(schema)
        assert "DOUBLE PRECISION" in result
        assert " REAL" not in result  # Space before to avoid matching "REAL" in other words
    
    def test_begin_transaction_removal(self):
        """Test BEGIN TRANSACTION removal."""
        schema = "BEGIN TRANSACTION;\nCREATE TABLE users (id int);\nCOMMIT;"
        result = convert_schema(schema)
        assert "BEGIN TRANSACTION" not in result
        assert "COMMIT" not in result
        assert "CREATE TABLE" in result
    
    def test_datetime_default_conversion(self):
        """Test DEFAULT DATETIME('now') → DEFAULT CURRENT_TIMESTAMP."""
        schema = 'CREATE TABLE users ("id" int, "created_at" DATETIME DEFAULT DATETIME(\'now\'));'
        result = convert_schema(schema)
        assert "DEFAULT CURRENT_TIMESTAMP" in result
        assert "DATETIME('now')" not in result
    
    def test_datetime_type_conversion(self):
        """Test DATETIME type → TIMESTAMP."""
        schema = 'CREATE TABLE users ("id" int, "created_at" DATETIME);'
        result = convert_schema(schema)
        assert "TIMESTAMP" in result
        # DATETIME should be converted to TIMESTAMP
    
    def test_empty_schema(self):
        """Test handling of empty schema."""
        result = convert_schema("")
        assert result == ""
    
    def test_schema_structure_preserved(self):
        """Test that overall schema structure is preserved."""
        schema = '''CREATE TABLE "users" (
"id" int,
"name" TEXT,
PRIMARY KEY ("id")
);'''
        result = convert_schema(schema)
        assert "CREATE TABLE" in result
        assert '"users"' in result
        assert '"id"' in result
        assert '"name"' in result
        assert "PRIMARY KEY" in result


class TestDetectConversionPatterns:
    """Tests for detect_conversion_patterns function."""
    
    def test_detects_concatenation(self):
        """Test detection of concatenation operator."""
        patterns = detect_conversion_patterns("SELECT a || b FROM t")
        assert patterns["concatenation"] is True
    
    def test_detects_datetime_now(self):
        """Test detection of DATETIME('now')."""
        patterns = detect_conversion_patterns("SELECT DATETIME('now')")
        assert patterns["datetime_now"] is True
    
    def test_detects_autoincrement(self):
        """Test detection of AUTOINCREMENT."""
        patterns = detect_conversion_patterns("id INTEGER AUTOINCREMENT")
        assert patterns["autoincrement"] is True
    
    def test_detects_pragma(self):
        """Test detection of PRAGMA."""
        patterns = detect_conversion_patterns("PRAGMA foreign_keys = ON")
        assert patterns["pragma"] is True
    
    def test_no_patterns(self):
        """Test query with no SQLite-specific patterns."""
        patterns = detect_conversion_patterns("SELECT * FROM users")
        assert patterns["concatenation"] is False
        assert patterns["datetime_now"] is False
        assert patterns["autoincrement"] is False


class TestValidatePostgresSyntax:
    """Tests for validate_postgres_syntax function."""
    
    def test_valid_postgres_query(self):
        """Test that valid PostgreSQL query passes."""
        sql = "SELECT * FROM users WHERE created_at > CURRENT_TIMESTAMP"
        is_valid, issues = validate_postgres_syntax(sql)
        assert is_valid is True
        assert len(issues) == 0
    
    def test_detects_unconverted_datetime(self):
        """Test detection of unconverted DATETIME('now')."""
        sql = "SELECT * FROM users WHERE created_at > DATETIME('now')"
        is_valid, issues = validate_postgres_syntax(sql)
        assert is_valid is False
        assert any("DATETIME" in issue for issue in issues)
    
    def test_detects_unconverted_autoincrement(self):
        """Test detection of unconverted AUTOINCREMENT."""
        sql = "id INTEGER PRIMARY KEY AUTOINCREMENT"
        is_valid, issues = validate_postgres_syntax(sql)
        assert is_valid is False
        assert any("AUTOINCREMENT" in issue for issue in issues)
    
    def test_detects_pragma(self):
        """Test detection of PRAGMA statement."""
        sql = "PRAGMA foreign_keys = ON"
        is_valid, issues = validate_postgres_syntax(sql)
        assert is_valid is False
        assert any("PRAGMA" in issue for issue in issues)


class TestConvertExample:
    """Tests for convert_example function."""
    
    def test_converts_full_example(self):
        """Test end-to-end conversion of a Mistral example."""
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert PostgreSQL query generator."
                },
                {
                    "role": "user",
                    "content": "Database: test\n\nSchema:\nPRAGMA foreign_keys = ON;\nCREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT);\n\nQuestion: Get all users\n\nGenerate the SQL query:"
                },
                {
                    "role": "assistant",
                    "content": "SELECT * FROM users WHERE created_at > DATETIME('now');"
                }
            ]
        }
        
        result = convert_example(example)
        
        # Check structure preserved
        assert "messages" in result
        assert len(result["messages"]) == 3
        
        # Check SQL converted
        assert "CURRENT_TIMESTAMP" in result["messages"][2]["content"]
        assert "DATETIME('now')" not in result["messages"][2]["content"]
        
        # Check schema converted
        user_content = result["messages"][1]["content"]
        assert "PRAGMA" not in user_content
        assert "SERIAL PRIMARY KEY" in user_content
        assert "AUTOINCREMENT" not in user_content
    
    def test_preserves_system_message(self):
        """Test that system message is unchanged."""
        example = {
            "messages": [
                {"role": "system", "content": "System prompt here"},
                {"role": "user", "content": "Database: test\n\nSchema:\nCREATE TABLE t (id int);\n\nQuestion: Q\n\nGenerate the SQL query:"},
                {"role": "assistant", "content": "SELECT 1;"}
            ]
        }
        
        result = convert_example(example)
        assert result["messages"][0]["content"] == "System prompt here"
    
    def test_preserves_question(self):
        """Test that question in user message is unchanged."""
        question = "How many users are there?"
        example = {
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": f"Database: test\n\nSchema:\nCREATE TABLE t (id int);\n\nQuestion: {question}\n\nGenerate the SQL query:"},
                {"role": "assistant", "content": "SELECT COUNT(*) FROM users;"}
            ]
        }
        
        result = convert_example(example)
        assert question in result["messages"][1]["content"]


class TestOutputFiles:
    """Tests for output file validation."""
    
    def test_train_postgres_exists(self):
        """Test that training PostgreSQL file exists."""
        if not TRAIN_POSTGRES_JSONL.exists():
            pytest.skip("PostgreSQL training file not created yet")
        assert TRAIN_POSTGRES_JSONL.exists()
    
    def test_dev_postgres_exists(self):
        """Test that dev PostgreSQL file exists."""
        if not DEV_POSTGRES_JSONL.exists():
            pytest.skip("PostgreSQL dev file not created yet")
        assert DEV_POSTGRES_JSONL.exists()
    
    def test_line_count_matches(self):
        """Test that output has same line count as input."""
        if not TRAIN_POSTGRES_JSONL.exists() or not TRAIN_MISTRAL_JSONL.exists():
            pytest.skip("Files not created yet")
        
        with open(TRAIN_MISTRAL_JSONL, "r") as f:
            mistral_count = sum(1 for _ in f)
        
        with open(TRAIN_POSTGRES_JSONL, "r") as f:
            postgres_count = sum(1 for _ in f)
        
        assert mistral_count == postgres_count
    
    def test_valid_json_per_line(self):
        """Test that each line is valid JSON."""
        if not TRAIN_POSTGRES_JSONL.exists():
            pytest.skip("PostgreSQL training file not created yet")
        
        with open(TRAIN_POSTGRES_JSONL, "r") as f:
            for i, line in enumerate(f):
                if i >= 100:  # Check first 100 lines
                    break
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Line {i+1} is not valid JSON: {e}")
    
    def test_no_pragma_in_output(self):
        """Test that no PRAGMA statements remain in output."""
        if not TRAIN_POSTGRES_JSONL.exists():
            pytest.skip("PostgreSQL training file not created yet")
        
        with open(TRAIN_POSTGRES_JSONL, "r") as f:
            for i, line in enumerate(f):
                if i >= 100:
                    break
                example = json.loads(line)
                user_content = example["messages"][1]["content"]
                # PRAGMA should be removed from schema
                schema_section = user_content.split("Schema:")[1].split("Question:")[0] if "Schema:" in user_content else ""
                assert "PRAGMA" not in schema_section, f"PRAGMA found in line {i+1}"
    
    def test_structure_preserved(self):
        """Test that message structure is preserved."""
        if not TRAIN_POSTGRES_JSONL.exists():
            pytest.skip("PostgreSQL training file not created yet")
        
        with open(TRAIN_POSTGRES_JSONL, "r") as f:
            example = json.loads(f.readline())
        
        assert "messages" in example
        assert len(example["messages"]) == 3
        
        roles = [msg["role"] for msg in example["messages"]]
        assert roles == ["system", "user", "assistant"]
        
        for msg in example["messages"]:
            assert "content" in msg
            assert len(msg["content"]) > 0
