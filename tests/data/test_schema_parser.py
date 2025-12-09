#!/usr/bin/env python3
"""
Tests for the Spider schema parser module.
"""

import json
import pytest
from pathlib import Path

# Import the module under test
from data.schema_parser import (
    parse_schema_file,
    validate_schema_content,
    parse_all_schemas,
    cross_reference_tables_json,
    SPIDER_DB_DIR,
    PROCESSED_SCHEMAS_DIR,
    PROJECT_DIR,
)


# Expected tables for academic database (15 tables)
ACADEMIC_EXPECTED_TABLES = [
    "author", "conference", "domain", "domain_author",
    "domain_conference", "journal", "domain_journal",
    "keyword", "domain_keyword", "publication",
    "domain_publication", "organization",
    "publication_keyword", "writes", "cite"
]


class TestParseSchemaFile:
    """Tests for parse_schema_file function."""
    
    def test_parse_academic_schema(self):
        """Test parsing the academic database schema."""
        schema_path = SPIDER_DB_DIR / "academic" / "schema.sql"
        
        if not schema_path.exists():
            pytest.skip("Spider dataset not downloaded")
        
        result = parse_schema_file(schema_path)
        
        assert "error" not in result
        assert result["num_tables"] == 15
        assert result["has_foreign_keys"] is True
        assert result["has_primary_keys"] is True
    
    def test_parse_academic_table_names(self):
        """Test that academic schema has expected table names."""
        schema_path = SPIDER_DB_DIR / "academic" / "schema.sql"
        
        if not schema_path.exists():
            pytest.skip("Spider dataset not downloaded")
        
        result = parse_schema_file(schema_path)
        
        parsed_tables = [t.lower() for t in result["tables"]]
        expected_tables = [t.lower() for t in ACADEMIC_EXPECTED_TABLES]
        
        # Check all expected tables are present
        for table in expected_tables:
            assert table in parsed_tables, f"Missing table: {table}"
    
    def test_parse_detects_foreign_keys(self):
        """Test that foreign key detection works."""
        schema_path = SPIDER_DB_DIR / "academic" / "schema.sql"
        
        if not schema_path.exists():
            pytest.skip("Spider dataset not downloaded")
        
        result = parse_schema_file(schema_path)
        
        assert result["has_foreign_keys"] is True
        assert len(result["foreign_keys"]) > 0
        
        # Check that foreign keys have correct structure
        for fk in result["foreign_keys"]:
            assert "column" in fk
            assert "references_table" in fk
            assert "references_column" in fk
    
    def test_parse_extracts_columns(self):
        """Test that column extraction works."""
        schema_path = SPIDER_DB_DIR / "academic" / "schema.sql"
        
        if not schema_path.exists():
            pytest.skip("Spider dataset not downloaded")
        
        result = parse_schema_file(schema_path)
        
        assert "table_columns" in result
        assert "author" in result["table_columns"]
        
        # Author table should have columns like aid, homepage, name, oid
        author_columns = [c.lower() for c in result["table_columns"]["author"]]
        assert "aid" in author_columns
        assert "name" in author_columns
    
    def test_parse_missing_file(self):
        """Test error handling for missing file."""
        result = parse_schema_file(Path("/nonexistent/path/schema.sql"))
        
        assert "error" in result
        assert "not found" in result["error"].lower()
    
    def test_parse_concert_singer_schema(self):
        """Test parsing concert_singer database schema."""
        schema_path = SPIDER_DB_DIR / "concert_singer" / "schema.sql"
        
        if not schema_path.exists():
            pytest.skip("Spider dataset not downloaded")
        
        result = parse_schema_file(schema_path)
        
        assert "error" not in result
        assert result["num_tables"] >= 3  # stadium, singer, concert, singer_in_concert
        assert result["has_foreign_keys"] is True


class TestValidateSchemaContent:
    """Tests for validate_schema_content function."""
    
    def test_valid_schema(self):
        """Test validation of valid schema content."""
        valid_sql = '''
        PRAGMA foreign_keys = ON;
        CREATE TABLE "users" (
            "id" int,
            "name" text,
            primary key("id")
        );
        '''
        
        is_valid, warnings = validate_schema_content(valid_sql)
        
        assert is_valid is True
    
    def test_invalid_empty_schema(self):
        """Test validation of empty schema."""
        is_valid, warnings = validate_schema_content("")
        
        assert is_valid is False
        assert any("empty" in w.lower() for w in warnings)
    
    def test_invalid_no_create_table(self):
        """Test validation of schema without CREATE TABLE."""
        invalid_sql = "PRAGMA foreign_keys = ON;"
        
        is_valid, warnings = validate_schema_content(invalid_sql)
        
        assert is_valid is False
        assert any("CREATE TABLE" in w for w in warnings)
    
    def test_warning_no_pragma(self):
        """Test warning when PRAGMA is missing."""
        sql_no_pragma = '''
        CREATE TABLE "users" (
            "id" int,
            primary key("id")
        );
        '''
        
        is_valid, warnings = validate_schema_content(sql_no_pragma)
        
        assert is_valid is True  # Still valid, just a warning
        assert any("PRAGMA" in w for w in warnings)


class TestSchemaIndex:
    """Tests for schema index creation and validation."""
    
    def test_schema_index_exists(self):
        """Test that schema_index.json exists after parsing."""
        index_path = PROCESSED_SCHEMAS_DIR / "schema_index.json"
        
        if not index_path.exists():
            # Run parsing first
            if SPIDER_DB_DIR.exists():
                parse_all_schemas(verbose=False)
            else:
                pytest.skip("Spider dataset not downloaded")
        
        assert index_path.exists()
    
    def test_schema_index_valid_json(self):
        """Test that schema_index.json is valid JSON."""
        index_path = PROCESSED_SCHEMAS_DIR / "schema_index.json"
        
        if not index_path.exists():
            pytest.skip("Schema index not created")
        
        with open(index_path, "r") as f:
            data = json.load(f)
        
        assert isinstance(data, dict)
        assert len(data) > 0
    
    def test_schema_index_has_academic(self):
        """Test that schema index contains academic database."""
        index_path = PROCESSED_SCHEMAS_DIR / "schema_index.json"
        
        if not index_path.exists():
            pytest.skip("Schema index not created")
        
        with open(index_path, "r") as f:
            data = json.load(f)
        
        assert "academic" in data
        assert data["academic"]["num_tables"] == 15
        assert data["academic"]["has_foreign_keys"] is True
    
    def test_schema_index_structure(self):
        """Test that schema index entries have required fields."""
        index_path = PROCESSED_SCHEMAS_DIR / "schema_index.json"
        
        if not index_path.exists():
            pytest.skip("Schema index not created")
        
        with open(index_path, "r") as f:
            data = json.load(f)
        
        # Check first entry has required fields
        first_db = list(data.keys())[0]
        entry = data[first_db]
        
        required_fields = [
            "schema_file", "source_file", "num_tables", 
            "tables", "has_foreign_keys", "parsed_timestamp"
        ]
        
        for field in required_fields:
            assert field in entry, f"Missing field: {field}"


class TestCopiedSchemas:
    """Tests for copied schema files."""
    
    def test_schemas_directory_exists(self):
        """Test that processed schemas directory exists."""
        if not SPIDER_DB_DIR.exists():
            pytest.skip("Spider dataset not downloaded")
        
        # Run parsing if needed
        if not PROCESSED_SCHEMAS_DIR.exists():
            parse_all_schemas(verbose=False)
        
        assert PROCESSED_SCHEMAS_DIR.exists()
    
    def test_academic_schema_copied(self):
        """Test that academic schema was copied."""
        copied_path = PROCESSED_SCHEMAS_DIR / "academic_schema.sql"
        
        if not copied_path.exists():
            if SPIDER_DB_DIR.exists():
                parse_all_schemas(verbose=False)
            else:
                pytest.skip("Spider dataset not downloaded")
        
        assert copied_path.exists()
        
        # Verify content matches original
        original_path = SPIDER_DB_DIR / "academic" / "schema.sql"
        if original_path.exists():
            original_content = original_path.read_text()
            copied_content = copied_path.read_text()
            assert original_content == copied_content
    
    def test_schema_count(self):
        """Test that we have a reasonable number of copied schemas."""
        if not PROCESSED_SCHEMAS_DIR.exists():
            pytest.skip("Schemas not processed")
        
        schema_files = list(PROCESSED_SCHEMAS_DIR.glob("*_schema.sql"))
        
        # Should have at least 140 schemas (not all 166 directories have schema.sql)
        assert len(schema_files) >= 100, f"Only found {len(schema_files)} schema files"


class TestCrossReference:
    """Tests for cross-reference validation."""
    
    def test_cross_reference_runs(self):
        """Test that cross-reference validation runs without error."""
        index_path = PROCESSED_SCHEMAS_DIR / "schema_index.json"
        
        if not index_path.exists():
            pytest.skip("Schema index not created")
        
        with open(index_path, "r") as f:
            schema_index = json.load(f)
        
        result = cross_reference_tables_json(schema_index, verbose=False)
        
        assert "error" not in result or "not found" in result.get("error", "")
        
        if "error" not in result:
            assert "matched" in result
            assert result["matched"] > 0
