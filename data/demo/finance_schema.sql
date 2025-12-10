-- Finance Database Schema
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
