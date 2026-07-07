-- schema.sql
-- Run this against the 'soc_db' database.

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    attack VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    severity VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS incidents (
    id SERIAL PRIMARY KEY,
    attack VARCHAR(100) NOT NULL,
    investigation TEXT,
    response TEXT,
    report TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_logs (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    action TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexing for faster queries on common lookup fields
CREATE INDEX idx_alerts_time ON alerts(time);
CREATE INDEX idx_alerts_attack ON alerts(attack);
CREATE INDEX idx_incidents_attack ON incidents(attack);
CREATE INDEX idx_agent_logs_timestamp ON agent_logs(timestamp);
