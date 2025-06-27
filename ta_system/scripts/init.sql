-- Initialize TimescaleDB for TA System

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create OHLCV data table
CREATE TABLE IF NOT EXISTS ohlcv_data (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open DECIMAL(18,8) NOT NULL,
    high DECIMAL(18,8) NOT NULL,
    low DECIMAL(18,8) NOT NULL,
    close DECIMAL(18,8) NOT NULL,
    volume BIGINT NOT NULL,
    PRIMARY KEY (time, symbol, timeframe)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('ohlcv_data', 'time', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indicator values table
CREATE TABLE IF NOT EXISTS indicator_values (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    indicator TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    value DECIMAL(18,8) NOT NULL,
    parameters JSONB,
    components JSONB,
    PRIMARY KEY (time, symbol, indicator, timeframe)
);

-- Create hypertable for indicator values
SELECT create_hypertable('indicator_values', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create trading signals table
CREATE TABLE IF NOT EXISTS trading_signals (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    strategy TEXT NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    price DECIMAL(18,8) NOT NULL,
    indicators JSONB,
    metadata JSONB,
    PRIMARY KEY (time, symbol, strategy)
);

-- Create hypertable for signals
SELECT create_hypertable('trading_signals', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv_data (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_indicator_symbol_indicator ON indicator_values (symbol, indicator, time DESC);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_strategy ON trading_signals (symbol, strategy, time DESC);

-- Create continuous aggregates for common queries
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_ohlcv
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    symbol,
    first(open, time) as open,
    max(high) as high,
    min(low) as low,
    last(close, time) as close,
    sum(volume) as volume
FROM ohlcv_data
WHERE timeframe = '1m'
GROUP BY day, symbol
WITH NO DATA;

-- Add refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('daily_ohlcv',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Create data retention policies
SELECT add_retention_policy('ohlcv_data', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('indicator_values', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('trading_signals', INTERVAL '5 years', if_not_exists => TRUE);

-- Create user roles
CREATE ROLE IF NOT EXISTS ta_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ta_readonly;

CREATE ROLE IF NOT EXISTS ta_readwrite;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ta_readwrite;

-- Grant permissions to application user
GRANT ta_readwrite TO ta_user;