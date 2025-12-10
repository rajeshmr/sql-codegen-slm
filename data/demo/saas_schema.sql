-- SaaS Platform Database Schema
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
