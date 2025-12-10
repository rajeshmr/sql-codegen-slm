-- Retail Chain Database Schema
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
