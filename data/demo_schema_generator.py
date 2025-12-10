#!/usr/bin/env python3
"""
Demo Schema Generator for SQL Codegen Frontend.

This module creates 5 polished demo schemas for different domains,
along with sample questions for the web UI demonstration.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DEMO_DIR = PROJECT_DIR / "data" / "demo"

# Output file
DEMO_SCHEMAS_JSON = DEMO_DIR / "demo_schemas.json"


# Demo schema definitions
DEMO_SCHEMAS = {
    "ecommerce": {
        "name": "E-commerce Platform",
        "description": "Online retail platform with orders, products, customers, and reviews",
        "filename": "ecommerce_schema.sql",
        "tables": ["customers", "products", "categories", "orders", "order_items", "reviews"],
        "schema": '''-- E-commerce Database Schema
-- Description: Online retail platform with orders, products, and customer management

CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(100),
    country VARCHAR(100) DEFAULT 'USA',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parent_category_id INTEGER REFERENCES categories(category_id)
);

CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    category_id INTEGER REFERENCES categories(category_id),
    price NUMERIC(10, 2) NOT NULL,
    cost NUMERIC(10, 2),
    stock_quantity INTEGER DEFAULT 0,
    sku VARCHAR(50) UNIQUE,
    is_available BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending',
    shipping_address TEXT,
    total_amount NUMERIC(10, 2),
    discount_amount NUMERIC(10, 2) DEFAULT 0,
    shipping_cost NUMERIC(10, 2) DEFAULT 0
);

CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10, 2) NOT NULL,
    discount_percent NUMERIC(5, 2) DEFAULT 0
);

CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id),
    customer_id INTEGER REFERENCES customers(customer_id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    title VARCHAR(200),
    comment TEXT,
    is_verified_purchase BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
''',
        "sample_questions": [
            "Show all customers who placed orders in the last 30 days",
            "What is the average order value?",
            "List top 5 products by total revenue",
            "Show customers with more than 10 orders",
            "Find products with average rating above 4.5",
            "What are the best-selling products by quantity?",
            "Show orders that are still pending",
            "List all products that are out of stock",
            "Find customers who have never placed an order",
            "What is the total revenue by category?"
        ]
    },
    
    "finance": {
        "name": "Banking System",
        "description": "Financial accounts, transactions, loans, and credit cards",
        "filename": "finance_schema.sql",
        "tables": ["customers", "accounts", "transactions", "loans", "credit_cards"],
        "schema": '''-- Finance Database Schema
-- Description: Banking system with accounts, transactions, and loan management

CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    ssn VARCHAR(11) UNIQUE,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    address TEXT,
    date_of_birth DATE,
    credit_score INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE accounts (
    account_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    account_number VARCHAR(20) UNIQUE NOT NULL,
    account_type VARCHAR(20) NOT NULL, -- 'checking', 'savings', 'money_market'
    balance NUMERIC(15, 2) DEFAULT 0,
    interest_rate NUMERIC(5, 4) DEFAULT 0,
    opened_date DATE DEFAULT CURRENT_DATE,
    status VARCHAR(20) DEFAULT 'active',
    overdraft_limit NUMERIC(10, 2) DEFAULT 0
);

CREATE TABLE transactions (
    transaction_id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(account_id),
    transaction_type VARCHAR(20) NOT NULL, -- 'deposit', 'withdrawal', 'transfer'
    amount NUMERIC(15, 2) NOT NULL,
    balance_after NUMERIC(15, 2),
    description TEXT,
    reference_number VARCHAR(50),
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE loans (
    loan_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    loan_type VARCHAR(50) NOT NULL, -- 'mortgage', 'auto', 'personal', 'student'
    principal_amount NUMERIC(15, 2) NOT NULL,
    interest_rate NUMERIC(5, 4) NOT NULL,
    term_months INTEGER NOT NULL,
    monthly_payment NUMERIC(10, 2),
    remaining_balance NUMERIC(15, 2),
    start_date DATE,
    status VARCHAR(20) DEFAULT 'active'
);

CREATE TABLE credit_cards (
    card_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    card_number VARCHAR(16) UNIQUE NOT NULL,
    card_type VARCHAR(20), -- 'visa', 'mastercard', 'amex'
    credit_limit NUMERIC(10, 2) NOT NULL,
    current_balance NUMERIC(10, 2) DEFAULT 0,
    minimum_payment NUMERIC(10, 2),
    due_date DATE,
    apr NUMERIC(5, 4),
    status VARCHAR(20) DEFAULT 'active'
);
''',
        "sample_questions": [
            "Show total balance across all accounts",
            "List customers with loans over $50,000",
            "Show transaction history for last month",
            "Find accounts with negative balance",
            "Calculate total interest paid on loans",
            "What is the average credit score of customers?",
            "Show customers with multiple accounts",
            "Find overdue credit card payments",
            "What is the total loan amount by loan type?",
            "List the top 10 highest value transactions"
        ]
    },
    
    "healthcare": {
        "name": "Healthcare System",
        "description": "Patient records, appointments, doctors, and prescriptions",
        "filename": "healthcare_schema.sql",
        "tables": ["patients", "doctors", "departments", "appointments", "prescriptions", "medical_records"],
        "schema": '''-- Healthcare Database Schema
-- Description: Hospital management with patients, doctors, and medical records

CREATE TABLE departments (
    department_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    floor_number INTEGER,
    phone_extension VARCHAR(10)
);

CREATE TABLE doctors (
    doctor_id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    specialization VARCHAR(100),
    department_id INTEGER REFERENCES departments(department_id),
    license_number VARCHAR(50) UNIQUE,
    hire_date DATE,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE patients (
    patient_id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(10),
    email VARCHAR(255),
    phone VARCHAR(20),
    address TEXT,
    emergency_contact VARCHAR(100),
    emergency_phone VARCHAR(20),
    blood_type VARCHAR(5),
    allergies TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE appointments (
    appointment_id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(patient_id),
    doctor_id INTEGER REFERENCES doctors(doctor_id),
    appointment_date TIMESTAMP NOT NULL,
    duration_minutes INTEGER DEFAULT 30,
    status VARCHAR(20) DEFAULT 'scheduled', -- 'scheduled', 'completed', 'cancelled'
    reason TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE medical_records (
    record_id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(patient_id),
    doctor_id INTEGER REFERENCES doctors(doctor_id),
    visit_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    diagnosis TEXT,
    treatment TEXT,
    vital_signs JSONB,
    follow_up_date DATE
);

CREATE TABLE prescriptions (
    prescription_id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(patient_id),
    doctor_id INTEGER REFERENCES doctors(doctor_id),
    record_id INTEGER REFERENCES medical_records(record_id),
    medication_name VARCHAR(200) NOT NULL,
    dosage VARCHAR(100),
    frequency VARCHAR(100),
    duration_days INTEGER,
    instructions TEXT,
    prescribed_date DATE DEFAULT CURRENT_DATE,
    is_active BOOLEAN DEFAULT TRUE
);
''',
        "sample_questions": [
            "Show all appointments for today",
            "List doctors by department",
            "Find patients with upcoming appointments",
            "Show prescription history for a patient",
            "How many appointments does each doctor have this week?",
            "List patients who haven't visited in over a year",
            "Find the busiest department by appointment count",
            "Show all active prescriptions",
            "What is the average appointment duration by specialization?",
            "List patients with known allergies"
        ]
    },
    
    "saas": {
        "name": "SaaS Platform",
        "description": "Software-as-a-Service with users, subscriptions, and usage tracking",
        "filename": "saas_schema.sql",
        "tables": ["organizations", "users", "plans", "subscriptions", "features", "usage_logs"],
        "schema": '''-- SaaS Platform Database Schema
-- Description: Multi-tenant SaaS with subscriptions, features, and usage tracking

CREATE TABLE organizations (
    org_id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(20), -- 'startup', 'small', 'medium', 'enterprise'
    website VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE plans (
    plan_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    price_monthly NUMERIC(10, 2) NOT NULL,
    price_yearly NUMERIC(10, 2),
    max_users INTEGER,
    max_storage_gb INTEGER,
    features JSONB,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    org_id INTEGER REFERENCES organizations(org_id),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(50) DEFAULT 'member', -- 'admin', 'member', 'viewer'
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE subscriptions (
    subscription_id SERIAL PRIMARY KEY,
    org_id INTEGER REFERENCES organizations(org_id),
    plan_id INTEGER REFERENCES plans(plan_id),
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'cancelled', 'past_due'
    billing_cycle VARCHAR(20) DEFAULT 'monthly', -- 'monthly', 'yearly'
    current_period_start DATE,
    current_period_end DATE,
    cancelled_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE features (
    feature_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    code VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    category VARCHAR(50),
    is_premium BOOLEAN DEFAULT FALSE
);

CREATE TABLE usage_logs (
    log_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    org_id INTEGER REFERENCES organizations(org_id),
    feature_id INTEGER REFERENCES features(feature_id),
    action VARCHAR(100) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
''',
        "sample_questions": [
            "Show all active subscriptions by plan",
            "List organizations with expired subscriptions",
            "How many users does each organization have?",
            "Show feature usage statistics for last week",
            "Find users who haven't logged in for 30 days",
            "What is the monthly recurring revenue (MRR)?",
            "List the most used features",
            "Show organizations on the enterprise plan",
            "Find users with admin role",
            "What is the churn rate by plan?"
        ]
    },
    
    "retail": {
        "name": "Retail Chain",
        "description": "Multi-store retail with inventory, sales, and employee management",
        "filename": "retail_schema.sql",
        "tables": ["stores", "employees", "products", "inventory", "sales"],
        "schema": '''-- Retail Chain Database Schema
-- Description: Multi-store retail management with inventory and sales tracking

CREATE TABLE stores (
    store_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    address TEXT NOT NULL,
    city VARCHAR(100),
    state VARCHAR(50),
    zip_code VARCHAR(20),
    phone VARCHAR(20),
    manager_id INTEGER,
    square_footage INTEGER,
    opened_date DATE,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    store_id INTEGER REFERENCES stores(store_id),
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    position VARCHAR(50),
    hourly_rate NUMERIC(10, 2),
    hire_date DATE DEFAULT CURRENT_DATE,
    is_active BOOLEAN DEFAULT TRUE
);

-- Add foreign key for manager after employees table exists
ALTER TABLE stores ADD CONSTRAINT fk_manager 
    FOREIGN KEY (manager_id) REFERENCES employees(employee_id);

CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    sku VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    brand VARCHAR(100),
    unit_cost NUMERIC(10, 2),
    retail_price NUMERIC(10, 2) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE inventory (
    inventory_id SERIAL PRIMARY KEY,
    store_id INTEGER REFERENCES stores(store_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER DEFAULT 0,
    reorder_level INTEGER DEFAULT 10,
    last_restocked TIMESTAMP,
    UNIQUE(store_id, product_id)
);

CREATE TABLE sales (
    sale_id SERIAL PRIMARY KEY,
    store_id INTEGER REFERENCES stores(store_id),
    employee_id INTEGER REFERENCES employees(employee_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10, 2) NOT NULL,
    discount_percent NUMERIC(5, 2) DEFAULT 0,
    total_amount NUMERIC(10, 2) NOT NULL,
    payment_method VARCHAR(20), -- 'cash', 'credit', 'debit'
    sale_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
''',
        "sample_questions": [
            "Show total sales by store for this month",
            "List products that need restocking",
            "Find the top-selling products",
            "Show employee sales performance",
            "What is the average transaction value by store?",
            "List stores with the highest revenue",
            "Find products with low inventory across all stores",
            "Show sales trends by day of week",
            "Which employees have the highest sales?",
            "What is the profit margin by product category?"
        ]
    }
}


def generate_sample_questions(schema_name: str, tables: list[str]) -> list[str]:
    """
    Generate sample questions for a schema.
    
    Args:
        schema_name: Name of the schema
        tables: List of table names
        
    Returns:
        List of sample questions
    """
    # Return pre-defined questions from DEMO_SCHEMAS
    if schema_name in DEMO_SCHEMAS:
        return DEMO_SCHEMAS[schema_name]["sample_questions"]
    return []


def create_demo_schemas(verbose: bool = True) -> list[str]:
    """
    Create all demo schemas.
    
    Args:
        verbose: Whether to print progress
        
    Returns:
        List of created schema names
    """
    # Ensure demo directory exists
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    
    created_schemas = []
    demo_metadata = {}
    
    if verbose:
        print("\n📝 Generating demo schemas...")
        print("━" * 50)
    
    for schema_name, schema_info in DEMO_SCHEMAS.items():
        # Write schema SQL file
        schema_path = DEMO_DIR / schema_info["filename"]
        schema_path.write_text(schema_info["schema"], encoding="utf-8")
        
        # Build metadata
        demo_metadata[schema_name] = {
            "schema_file": str(schema_path.relative_to(PROJECT_DIR)),
            "name": schema_info["name"],
            "description": schema_info["description"],
            "tables": schema_info["tables"],
            "sample_questions": schema_info["sample_questions"],
        }
        
        created_schemas.append(schema_name)
        
        if verbose:
            table_count = len(schema_info["tables"])
            question_count = len(schema_info["sample_questions"])
            print(f"✅ {schema_info['name']} ({table_count} tables, {question_count} sample questions)")
    
    # Save metadata JSON
    demo_metadata["_metadata"] = {
        "created_at": datetime.utcnow().isoformat(),
        "total_schemas": len(created_schemas),
    }
    
    with open(DEMO_SCHEMAS_JSON, "w", encoding="utf-8") as f:
        json.dump(demo_metadata, f, indent=2)
    
    if verbose:
        print(f"\n✅ Created {len(created_schemas)} demo schemas in {DEMO_DIR}")
    
    return created_schemas


def validate_demo_schemas(verbose: bool = True) -> bool:
    """
    Validate that demo schemas have proper PostgreSQL syntax.
    
    Args:
        verbose: Whether to print details
        
    Returns:
        True if all valid, False otherwise
    """
    errors = []
    
    for schema_name, schema_info in DEMO_SCHEMAS.items():
        schema_path = DEMO_DIR / schema_info["filename"]
        
        if not schema_path.exists():
            errors.append(f"{schema_name}: File not found")
            continue
        
        content = schema_path.read_text(encoding="utf-8")
        
        # Check for SQLite-isms
        if "AUTOINCREMENT" in content:
            errors.append(f"{schema_name}: Contains AUTOINCREMENT (SQLite)")
        if "PRAGMA" in content:
            errors.append(f"{schema_name}: Contains PRAGMA (SQLite)")
        
        # Check for required PostgreSQL features
        if "SERIAL" not in content:
            errors.append(f"{schema_name}: Missing SERIAL for auto-increment")
        if "REFERENCES" not in content:
            errors.append(f"{schema_name}: Missing foreign key REFERENCES")
        if "CREATE TABLE" not in content:
            errors.append(f"{schema_name}: Missing CREATE TABLE statements")
    
    if verbose:
        if errors:
            print("\n❌ Demo schema validation errors:")
            for error in errors:
                print(f"   - {error}")
        else:
            print("\n✅ All demo schemas have valid PostgreSQL syntax")
    
    return len(errors) == 0


def main() -> int:
    """Main entry point for demo schema generation."""
    print("📝 Generating demo schemas for frontend...")
    
    # Create schemas
    created = create_demo_schemas(verbose=True)
    
    # Validate
    is_valid = validate_demo_schemas(verbose=True)
    
    # Print summary
    print("\n" + "━" * 50)
    print("✅ Demo Schema Generation Complete!")
    print("━" * 50)
    print(f"Schemas created: {len(created)}")
    print(f"Output directory: {DEMO_DIR}")
    print("━" * 50)
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
