# SQLite to PostgreSQL Conversion Reference

This document describes the conversion rules applied when transforming SQLite SQL syntax to PostgreSQL-compatible syntax for the training data.

## Overview

The Spider dataset uses SQLite databases. Since our model targets PostgreSQL, we convert SQLite-specific syntax to PostgreSQL equivalents while preserving query semantics.

## Data Types

### AUTOINCREMENT → SERIAL

SQLite uses `AUTOINCREMENT` for auto-incrementing columns, PostgreSQL uses `SERIAL`.

**Before (SQLite):**
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
);
```

**After (PostgreSQL):**
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT
);
```

### BLOB → BYTEA

SQLite's `BLOB` type maps to PostgreSQL's `BYTEA`.

**Before (SQLite):**
```sql
CREATE TABLE files (
    id INTEGER,
    data BLOB
);
```

**After (PostgreSQL):**
```sql
CREATE TABLE files (
    id INTEGER,
    data BYTEA
);
```

### REAL → DOUBLE PRECISION

SQLite's `REAL` type maps to PostgreSQL's `DOUBLE PRECISION` for floating-point numbers.

**Before (SQLite):**
```sql
CREATE TABLE measurements (
    id INTEGER,
    value REAL
);
```

**After (PostgreSQL):**
```sql
CREATE TABLE measurements (
    id INTEGER,
    value DOUBLE PRECISION
);
```

### DATETIME → TIMESTAMP

SQLite's `DATETIME` type maps to PostgreSQL's `TIMESTAMP`.

**Before (SQLite):**
```sql
CREATE TABLE events (
    id INTEGER,
    created_at DATETIME
);
```

**After (PostgreSQL):**
```sql
CREATE TABLE events (
    id INTEGER,
    created_at TIMESTAMP
);
```

## Date/Time Functions

### DATETIME('now') → CURRENT_TIMESTAMP

**Before (SQLite):**
```sql
SELECT * FROM users WHERE created_at > DATETIME('now');
```

**After (PostgreSQL):**
```sql
SELECT * FROM users WHERE created_at > CURRENT_TIMESTAMP;
```

### DATE('now') → CURRENT_DATE

**Before (SQLite):**
```sql
SELECT * FROM users WHERE birth_date = DATE('now');
```

**After (PostgreSQL):**
```sql
SELECT * FROM users WHERE birth_date = CURRENT_DATE;
```

### TIME('now') → CURRENT_TIME

**Before (SQLite):**
```sql
SELECT * FROM logs WHERE log_time = TIME('now');
```

**After (PostgreSQL):**
```sql
SELECT * FROM logs WHERE log_time = CURRENT_TIME;
```

### strftime() → TO_CHAR()

SQLite's `strftime()` function maps to PostgreSQL's `TO_CHAR()` with different format specifiers.

**Format String Conversions:**
| SQLite | PostgreSQL | Description |
|--------|------------|-------------|
| `%Y` | `YYYY` | 4-digit year |
| `%m` | `MM` | 2-digit month |
| `%d` | `DD` | 2-digit day |
| `%H` | `HH24` | 24-hour hour |
| `%M` | `MI` | Minutes |
| `%S` | `SS` | Seconds |

**Before (SQLite):**
```sql
SELECT strftime('%Y-%m-%d', created_at) FROM users;
```

**After (PostgreSQL):**
```sql
SELECT TO_CHAR(created_at, 'YYYY-MM-DD') FROM users;
```

## String Functions

### SUBSTR() → SUBSTRING()

SQLite's `SUBSTR(string, start, length)` maps to PostgreSQL's `SUBSTRING(string FROM start FOR length)`.

**Before (SQLite):**
```sql
SELECT SUBSTR(name, 1, 3) FROM users;
```

**After (PostgreSQL):**
```sql
SELECT SUBSTRING(name FROM 1 FOR 3) FROM users;
```

### Concatenation (||)

Both SQLite and PostgreSQL support the `||` operator for string concatenation. **No conversion needed.**

```sql
-- Works in both SQLite and PostgreSQL
SELECT first_name || ' ' || last_name FROM users;
```

PostgreSQL also supports `CONCAT()` function, but `||` is kept as-is since it's compatible.

## Schema Statements

### PRAGMA Statements

SQLite's `PRAGMA` statements are removed as they have no PostgreSQL equivalent and are not needed.

**Before (SQLite):**
```sql
PRAGMA foreign_keys = ON;
CREATE TABLE users (id INTEGER);
```

**After (PostgreSQL):**
```sql
CREATE TABLE users (id INTEGER);
```

### Transaction Statements

`BEGIN TRANSACTION` and `COMMIT` statements are removed from schema definitions.

**Before (SQLite):**
```sql
BEGIN TRANSACTION;
CREATE TABLE users (id INTEGER);
COMMIT;
```

**After (PostgreSQL):**
```sql
CREATE TABLE users (id INTEGER);
```

## Compatible Syntax (No Conversion Needed)

The following SQL syntax is compatible between SQLite and PostgreSQL:

- **SELECT statements**: Basic SELECT, WHERE, ORDER BY, LIMIT
- **JOIN operations**: INNER JOIN, LEFT JOIN, RIGHT JOIN
- **Aggregations**: COUNT, SUM, AVG, MIN, MAX
- **GROUP BY and HAVING clauses**
- **Subqueries**
- **UNION, INTERSECT, EXCEPT**
- **Common string functions**: LENGTH, UPPER, LOWER, TRIM
- **Comparison operators**: =, <>, <, >, <=, >=
- **Logical operators**: AND, OR, NOT
- **NULL handling**: IS NULL, IS NOT NULL, COALESCE
- **CASE expressions**
- **DISTINCT keyword**
- **Aliases**: AS keyword for column and table aliases

## Edge Cases

### Boolean Values

SQLite uses `1` and `0` for boolean values, while PostgreSQL supports `TRUE` and `FALSE`. However, PostgreSQL also accepts `1` and `0` in boolean contexts, so no conversion is performed.

### Quote Styles

Both SQLite and PostgreSQL support double quotes for identifiers:
```sql
SELECT "column_name" FROM "table_name";
```

This syntax is kept as-is.

### INSERT Statements

INSERT statements in schema files (sample data) are kept as-is since the syntax is compatible.

## Validation

After conversion, the following checks are performed:

1. No `DATETIME('now')` patterns remain
2. No `DATE('now')` patterns remain
3. No `AUTOINCREMENT` keywords remain
4. No `PRAGMA` statements remain
5. JSON structure is preserved (messages array with system/user/assistant)

## Statistics

The converter tracks and reports:

- Number of AUTOINCREMENT → SERIAL conversions
- Number of PRAGMA statement removals
- Number of date function conversions
- Number of string function conversions
- Total examples processed and converted
