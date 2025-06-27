#!/bin/bash
# InfluxDB Retention Policies Setup Script
# Configures optimal retention policies for different data types and use cases

set -e

echo "⏰ Setting up InfluxDB retention policies for ML trading data..."

# Configuration
INFLUX_HOST="http://localhost:8086"
INFLUX_TOKEN="${DOCKER_INFLUXDB_INIT_ADMIN_TOKEN}"
INFLUX_ORG="${DOCKER_INFLUXDB_INIT_ORG}"

# Function to update bucket retention policy
update_retention_policy() {
    local bucket_name="$1"
    local retention_period="$2"
    local description="$3"
    
    echo "⏳ Updating retention policy for bucket: $bucket_name"
    echo "   Retention: $retention_period"
    echo "   Purpose: $description"
    
    # Update bucket retention
    influx bucket update \
        --host "$INFLUX_HOST" \
        --token "$INFLUX_TOKEN" \
        --org "$INFLUX_ORG" \
        --name "$bucket_name" \
        --retention "$retention_period" \
        --description "$description"
    
    echo "   ✅ Updated retention policy for '$bucket_name'"
}

echo "📊 Configuring optimal retention policies..."

# Core market data - permanent retention for backtesting
update_retention_policy "market_data" "0s" \
    "Primary market data with permanent retention for comprehensive backtesting"

# Crypto data - 5 years (good for long-term crypto analysis)
update_retention_policy "crypto_data" "43800h" \
    "Cryptocurrency data with 5-year retention for long-term analysis"

# Crypto order book - 1 year (high-frequency microstructure data)
update_retention_policy "crypto_orderbook" "8760h" \
    "Crypto order book data with 1-year retention for microstructure analysis"

# Equity data - 10 years (regulatory and long-term analysis)
update_retention_policy "equity_data" "87600h" \
    "Equity market data with 10-year retention for regulatory compliance"

# Equity fundamentals - permanent (slow-changing, valuable historical data)
update_retention_policy "equity_fundamentals" "0s" \
    "Fundamental data with permanent retention for factor investing"

# Futures data - 7 years (contract lifecycle and seasonality analysis)
update_retention_policy "futures_data" "61320h" \
    "Futures data with 7-year retention for contract lifecycle analysis"

# Options data - 3 years (sufficient for volatility surface modeling)
update_retention_policy "options_data" "26280h" \
    "Options data with 3-year retention for volatility surface modeling"

# ML features - 3 years (balance between storage and model training needs)
update_retention_policy "ml_features" "26280h" \
    "ML features with 3-year retention for model training and validation"

# Correlations - 5 years (important for regime analysis)
update_retention_policy "correlations" "43800h" \
    "Cross-asset correlations with 5-year retention for regime analysis"

# Strategy performance - permanent (critical business data)
update_retention_policy "strategy_performance" "0s" \
    "Strategy performance metrics with permanent retention for business analytics"

# Real-time data - 30 days (high-volume, short-term relevance)
update_retention_policy "realtime_data" "720h" \
    "Real-time tick data with 30-day retention for immediate analysis"

# HFT signals - 90 days (algorithm development and debugging)
update_retention_policy "hft_signals" "2160h" \
    "HFT signals with 90-day retention for algorithm development"

# System metrics - 1 year (operational monitoring)
update_retention_policy "system_metrics" "8760h" \
    "System metrics with 1-year retention for operational monitoring"

# Data quality - 1 year (quality trend analysis)
update_retention_policy "data_quality" "8760h" \
    "Data quality metrics with 1-year retention for trend analysis"

echo ""
echo "⚙️ Creating continuous queries for data lifecycle management..."

# Create task for automatic data cleanup of expired real-time data
create_cleanup_task() {
    local task_name="$1"
    local flux_script="$2"
    
    echo "🧹 Creating cleanup task: $task_name"
    
    # Check if task already exists
    if influx task list --host "$INFLUX_HOST" --token "$INFLUX_TOKEN" --org "$INFLUX_ORG" --name "$task_name" &>/dev/null; then
        echo "   Task '$task_name' already exists, skipping..."
        return 0
    fi
    
    # Create the task
    echo "$flux_script" | influx task create \
        --host "$INFLUX_HOST" \
        --token "$INFLUX_TOKEN" \
        --org "$INFLUX_ORG"
    
    echo "   ✅ Created cleanup task '$task_name'"
}

# Weekly cleanup task for orphaned data
create_cleanup_task "weekly_cleanup" '
option task = {name: "weekly_cleanup", every: 7d}

// Clean up any orphaned or corrupted data points
from(bucket: "realtime_data")
    |> range(start: -35d, stop: -30d)
    |> filter(fn: (r) => r._value == 0 or r._value < 0)
    |> drop()
'

# Monthly storage optimization task
create_cleanup_task "monthly_optimization" '
option task = {name: "monthly_optimization", every: 30d}

// Compact and optimize high-frequency data
from(bucket: "crypto_orderbook")
    |> range(start: -400d, stop: -365d)
    |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
    |> to(bucket: "crypto_data", orgID: "'${INFLUX_ORG}'")
'

echo ""
echo "📊 Storage optimization summary:"
echo ""
echo "🔄 Retention Policies:"
echo "   • market_data: Permanent (core trading data)"
echo "   • crypto_data: 5 years (long-term crypto analysis)"
echo "   • equity_data: 10 years (regulatory compliance)"
echo "   • futures_data: 7 years (contract lifecycle)"
echo "   • options_data: 3 years (volatility modeling)"
echo "   • ml_features: 3 years (model training)"
echo "   • strategy_performance: Permanent (business critical)"
echo "   • realtime_data: 30 days (high-volume, short-term)"
echo ""
echo "🧹 Cleanup Tasks:"
echo "   • weekly_cleanup: Remove invalid data points"
echo "   • monthly_optimization: Compact old high-frequency data"
echo ""
echo "💡 Storage Optimization Tips:"
echo "   • Use downsampling for older data"
echo "   • Archive strategy performance data externally"
echo "   • Monitor disk usage regularly"
echo "   • Consider cold storage for data older than retention period"
echo ""
echo "⚡ Estimated Storage Usage (per year):"
echo "   • 1-minute equity data: ~50GB/year"
echo "   • 1-minute crypto data: ~20GB/year"
echo "   • Tick data (if enabled): ~500GB/year"
echo "   • ML features: ~10GB/year"
echo ""
echo "🎉 Retention policies configured successfully!"