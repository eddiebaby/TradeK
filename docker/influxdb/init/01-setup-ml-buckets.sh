#!/bin/bash
# InfluxDB ML Buckets Setup Script
# Creates additional buckets and configurations for ML-ready trading data

set -e

echo "🤖 Setting up ML-ready InfluxDB buckets and configurations..."

# Configuration
INFLUX_HOST="http://localhost:8086"
INFLUX_TOKEN="${DOCKER_INFLUXDB_INIT_ADMIN_TOKEN}"
INFLUX_ORG="${DOCKER_INFLUXDB_INIT_ORG}"

# Wait for InfluxDB to be fully ready
echo "⏳ Waiting for InfluxDB to be ready..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s "$INFLUX_HOST/health" | grep -q '"status":"pass"'; then
        echo "✅ InfluxDB is ready!"
        break
    fi
    echo "   Attempt $((attempt + 1))/$max_attempts - waiting 2 seconds..."
    sleep 2
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ InfluxDB failed to become ready within expected time"
    exit 1
fi

# Function to create bucket if it doesn't exist
create_bucket() {
    local bucket_name="$1"
    local retention="$2"
    local description="$3"
    
    echo "📊 Creating bucket: $bucket_name"
    
    # Check if bucket already exists
    if influx bucket list --host "$INFLUX_HOST" --token "$INFLUX_TOKEN" --org "$INFLUX_ORG" --name "$bucket_name" &>/dev/null; then
        echo "   Bucket '$bucket_name' already exists, skipping..."
        return 0
    fi
    
    # Create the bucket
    influx bucket create \
        --host "$INFLUX_HOST" \
        --token "$INFLUX_TOKEN" \
        --org "$INFLUX_ORG" \
        --name "$bucket_name" \
        --retention "$retention" \
        --description "$description"
    
    echo "   ✅ Created bucket '$bucket_name' with retention $retention"
}

# Function to create task if it doesn't exist
create_task() {
    local task_name="$1"
    local flux_script="$2"
    
    echo "🔧 Creating task: $task_name"
    
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
    
    echo "   ✅ Created task '$task_name'"
}

# Create ML-specific buckets with appropriate retention policies
echo "🗄️ Creating ML-ready buckets..."

# Main market data bucket (already created during init, but verify)
create_bucket "market_data" "0s" "Primary market data for all asset classes"

# Crypto-specific buckets
create_bucket "crypto_data" "5y" "Cryptocurrency market data with extended retention"
create_bucket "crypto_orderbook" "1y" "Crypto order book data for microstructure analysis"

# Equity-specific buckets  
create_bucket "equity_data" "10y" "Equity market data with long-term retention"
create_bucket "equity_fundamentals" "0s" "Fundamental data for factor investing"

# Futures and derivatives
create_bucket "futures_data" "7y" "Futures market data with contract-specific retention"
create_bucket "options_data" "3y" "Options data with Greeks and volatility surfaces"

# ML and analytics buckets
create_bucket "ml_features" "3y" "Pre-computed ML features and indicators"
create_bucket "correlations" "5y" "Cross-asset correlations and relationships"
create_bucket "strategy_performance" "0s" "Strategy performance metrics and PnL"

# Real-time and high-frequency data
create_bucket "realtime_data" "30d" "Real-time tick data with short retention"
create_bucket "hft_signals" "90d" "High-frequency trading signals and microstructure"

# Monitoring and system metrics
create_bucket "system_metrics" "1y" "System performance and monitoring data"
create_bucket "data_quality" "1y" "Data quality metrics and validation results"

# Create downsampling tasks for performance optimization
echo "⚡ Creating downsampling tasks..."

# 5-minute downsampling task
create_task "downsample_5min" '
option task = {name: "downsample_5min", every: 5m}

from(bucket: "market_data")
    |> range(start: -10m)
    |> filter(fn: (r) => r._measurement =~ /.*_1min$/)
    |> aggregateWindow(every: 5m, fn: last, createEmpty: false)
    |> map(fn: (r) => ({r with _measurement: regex.replaceAllString(r: /1min/, v: r._measurement, t: "5min")}))
    |> to(bucket: "market_data", orgID: "'${INFLUX_ORG}'")
'

# Hourly downsampling task  
create_task "downsample_1h" '
option task = {name: "downsample_1h", every: 1h}

from(bucket: "market_data")
    |> range(start: -2h)
    |> filter(fn: (r) => r._measurement =~ /.*_5min$/)
    |> aggregateWindow(every: 1h, fn: last, createEmpty: false)  
    |> map(fn: (r) => ({r with _measurement: regex.replaceAllString(r: /5min/, v: r._measurement, t: "1h")}))
    |> to(bucket: "market_data", orgID: "'${INFLUX_ORG}'")
'

# Daily downsampling task
create_task "downsample_1d" '
option task = {name: "downsample_1d", every: 1d}

from(bucket: "market_data")
    |> range(start: -2d)
    |> filter(fn: (r) => r._measurement =~ /.*_1h$/)
    |> aggregateWindow(every: 1d, fn: last, createEmpty: false)
    |> map(fn: (r) => ({r with _measurement: regex.replaceAllString(r: /1h/, v: r._measurement, t: "1d")}))
    |> to(bucket: "market_data", orgID: "'${INFLUX_ORG}'")
'

# Data quality monitoring task
create_task "data_quality_monitor" '
option task = {name: "data_quality_monitor", every: 1h}

// Monitor data freshness
data_freshness = from(bucket: "market_data")
    |> range(start: -2h)
    |> group(columns: ["_measurement", "symbol"])
    |> max(column: "_time")
    |> map(fn: (r) => ({
        _time: now(),
        _measurement: "data_freshness",
        symbol: r.symbol,
        asset_class: r._measurement,
        last_update: r._time,
        age_minutes: float(v: uint(v: now()) - uint(v: r._time)) / 60000000000.0
    }))

data_freshness |> to(bucket: "data_quality", orgID: "'${INFLUX_ORG}'")
'

echo ""
echo "🎉 InfluxDB ML setup completed successfully!"
echo ""
echo "📊 Created buckets:"
echo "   • market_data (primary bucket)"
echo "   • crypto_data, crypto_orderbook"  
echo "   • equity_data, equity_fundamentals"
echo "   • futures_data, options_data"
echo "   • ml_features, correlations"
echo "   • strategy_performance"
echo "   • realtime_data, hft_signals"
echo "   • system_metrics, data_quality"
echo ""
echo "⚡ Created tasks:"
echo "   • downsample_5min (1-min → 5-min data)"
echo "   • downsample_1h (5-min → hourly data)"
echo "   • downsample_1d (hourly → daily data)"
echo "   • data_quality_monitor (freshness monitoring)"
echo ""
echo "🚀 Your InfluxDB is now optimized for ML trading workloads!"