---
title: Text-to-SQL Generator
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
suggested_hardware: t4-small
---

# Text-to-SQL Generator

Convert natural language questions into PostgreSQL queries using a fine-tuned Mistral-7B model.

## Model

This demo uses [rajeshmanikka/mistral-7b-text-to-sql](https://huggingface.co/rajeshmanikka/mistral-7b-text-to-sql), a LoRA fine-tuned version of Mistral-7B trained on the Spider dataset.

## Usage

1. **Enter your database schema** in the left text box (PostgreSQL CREATE TABLE statements)
2. **Ask a question** about your data in natural language
3. **Click "Generate SQL"** to get the query

## Features

- Generates PostgreSQL-compatible SQL
- Supports JOINs, aggregations, GROUP BY, ORDER BY, subqueries
- Uses 4-bit quantization for efficient inference
- Pre-loaded examples to try

## Limitations

- PostgreSQL syntax only (not MySQL or SQLite)
- Requires complete database schema as input
- First query may take 2-3 minutes as the model loads

## Links

- **Model**: [rajeshmanikka/mistral-7b-text-to-sql](https://huggingface.co/rajeshmanikka/mistral-7b-text-to-sql)
- **GitHub**: [sql-codegen-slm](https://github.com/rajeshmr/sql-codegen-slm)
- **Dataset**: [Spider](https://yale-lily.github.io/spider)

## License

Apache 2.0
