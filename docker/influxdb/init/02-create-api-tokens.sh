#!/bin/bash
# InfluxDB API Tokens Setup Script
# Creates specific API tokens for different services and access levels

set -e

echo "🔑 Setting up InfluxDB API tokens for ML services..."

# Configuration
INFLUX_HOST="http://localhost:8086"
INFLUX_TOKEN="${DOCKER_INFLUXDB_INIT_ADMIN_TOKEN}"
INFLUX_ORG="${DOCKER_INFLUXDB_INIT_ORG}"

# Function to create API token if it doesn't exist
create_api_token() {
    local token_name="$1"
    local description="$2"
    local permissions="$3"
    
    echo "🔐 Creating API token: $token_name"
    
    # Check if token already exists
    if influx auth list --host "$INFLUX_HOST" --token "$INFLUX_TOKEN" --org "$INFLUX_ORG" --user admin | grep -q "$token_name"; then
        echo "   Token '$token_name' already exists, skipping..."
        return 0
    fi
    
    # Create the token
    local created_token=$(influx auth create \
        --host "$INFLUX_HOST" \
        --token "$INFLUX_TOKEN" \
        --org "$INFLUX_ORG" \
        --description "$description" \
        $permissions \
        --json | jq -r '.token')
    
    echo "   ✅ Created token '$token_name'"
    echo "   📝 Token: $created_token"
    
    # Save token to file for later reference
    echo "$token_name=$created_token" >> /backups/api_tokens.env
    echo "   💾 Saved to /backups/api_tokens.env"
}

# Create backup directory for tokens
mkdir -p /backups
echo "# InfluxDB API Tokens - $(date)" > /backups/api_tokens.env
echo "# Use these tokens for different services and access levels" >> /backups/api_tokens.env
echo "" >> /backups/api_tokens.env

echo "🔑 Creating service-specific API tokens..."

# ML Backfill Service Token (read/write to all market data buckets)
create_api_token "ml-backfill-service" \
    "Token for ML backfill orchestrator - full access to market data buckets" \
    "--read-bucket market_data --write-bucket market_data --read-bucket crypto_data --write-bucket crypto_data --read-bucket equity_data --write-bucket equity_data --read-bucket futures_data --write-bucket futures_data --read-bucket options_data --write-bucket options_data"

# Strategy Execution Token (read market data, write performance data)
create_api_token "strategy-execution" \
    "Token for strategy execution - read market data, write performance metrics" \
    "--read-bucket market_data --read-bucket crypto_data --read-bucket equity_data --read-bucket futures_data --read-bucket ml_features --write-bucket strategy_performance"

# Real-time Trading Token (read/write real-time data)
create_api_token "realtime-trading" \
    "Token for real-time trading systems - access to live data feeds" \
    "--read-bucket market_data --write-bucket realtime_data --read-bucket realtime_data --write-bucket hft_signals --read-bucket hft_signals"

# ML Feature Engineering Token (read market data, write ML features)
create_api_token "ml-feature-engineering" \
    "Token for ML feature engineering - read raw data, write computed features" \
    "--read-bucket market_data --read-bucket crypto_data --read-bucket equity_data --write-bucket ml_features --write-bucket correlations"

# Read-only Analytics Token (read access to all buckets)
create_api_token "analytics-readonly" \
    "Read-only token for analytics and reporting" \
    "--read-bucket market_data --read-bucket crypto_data --read-bucket equity_data --read-bucket futures_data --read-bucket options_data --read-bucket ml_features --read-bucket correlations --read-bucket strategy_performance"

# Monitoring Token (read system metrics, write monitoring data)
create_api_token "monitoring-service" \
    "Token for monitoring and alerting systems" \
    "--read-bucket system_metrics --write-bucket system_metrics --read-bucket data_quality --write-bucket data_quality"

# Data Quality Token (read all data, write quality metrics)
create_api_token "data-quality-service" \
    "Token for data quality validation and monitoring" \
    "--read-bucket market_data --read-bucket crypto_data --read-bucket equity_data --write-bucket data_quality"

# Backup Token (read all buckets for backup purposes)
create_api_token "backup-service" \
    "Token for backup and disaster recovery operations" \
    "--read-bucket market_data --read-bucket crypto_data --read-bucket equity_data --read-bucket futures_data --read-bucket options_data --read-bucket ml_features --read-bucket correlations --read-bucket strategy_performance --read-bucket system_metrics --read-bucket data_quality"

echo ""
echo "🎉 API tokens setup completed!"
echo ""
echo "📝 Token summary:"
echo "   • ml-backfill-service: Full access to market data buckets"
echo "   • strategy-execution: Read market data, write performance"
echo "   • realtime-trading: Real-time data access"
echo "   • ml-feature-engineering: ML feature computation"
echo "   • analytics-readonly: Read-only analytics access"
echo "   • monitoring-service: System monitoring"
echo "   • data-quality-service: Data quality validation"
echo "   • backup-service: Backup and recovery"
echo ""
echo "💾 All tokens saved to: /backups/api_tokens.env"
echo "🔒 Keep these tokens secure and use appropriate tokens for each service"
echo ""
echo "📋 To use tokens in your applications:"
echo "   export INFLUXDB_TOKEN=\$(grep 'ml-backfill-service' /backups/api_tokens.env | cut -d'=' -f2)"