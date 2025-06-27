# 🐳 Dockerized InfluxDB Infrastructure - Complete Setup Guide

## 🎯 What This Provides

A **production-ready, persistent InfluxDB infrastructure** with comprehensive ML-optimized configuration:

- **🔄 Persistent Data Storage** with automatic backups and recovery
- **🤖 ML-Optimized Schema** with pre-configured buckets and retention policies
- **🔐 Security-First Configuration** with API tokens and authentication
- **📊 Multi-Environment Support** (development, staging, production)
- **⚡ Performance Tuning** for high-frequency trading workloads
- **📈 Monitoring Integration** with Prometheus and Grafana

---

## 🚀 Quick Start

### Option 1: Development Environment (Recommended for Testing)
```bash
# Start with development overrides
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Check InfluxDB status
docker-compose logs influxdb
```

### Option 2: Full Production Stack
```bash
# Copy and configure environment variables
cp .env.example .env
vim .env  # Configure your API keys and passwords

# Start full production stack
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Option 3: Basic Setup
```bash
# Start core services only
docker-compose up -d influxdb redis qdrant
```

---

## 🔧 Complete Setup Instructions

### Step 1: Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration file
vim .env
```

**Essential Configuration:**
```bash
# InfluxDB Authentication (use strong passwords in production)
INFLUXDB_ADMIN_USERNAME=admin
INFLUXDB_ADMIN_PASSWORD=your-secure-password
INFLUXDB_ADMIN_TOKEN=your-super-secret-token
INFLUXDB_ORG=tradeknowledge
INFLUXDB_BUCKET=market_data

# Market Data API Keys
POLYGON_API_KEY=your_polygon_key_here
KRAKEN_API_KEY=your_kraken_key_here
COINBASE_API_KEY=your_coinbase_key_here
```

### Step 2: Initialize Infrastructure

```bash
# Create necessary directories
mkdir -p data/influxdb-backups
mkdir -p logs
mkdir -p docker/influxdb/init

# Set proper permissions
chmod +x docker/influxdb/init/*.sh

# Start InfluxDB and core services
docker-compose up -d influxdb redis qdrant
```

### Step 3: Verify InfluxDB Setup

```bash
# Check InfluxDB health
curl -s http://localhost:8086/health

# Expected response: {"name":"influxdb","message":"ready for queries and writes","status":"pass"}

# View initialization logs
docker-compose logs influxdb | grep -E "(bucket|token|task)"

# Check created buckets
docker exec tradeknowledge-influxdb influx bucket list --host http://localhost:8086 --token tradeknowledge-super-secret-token --org tradeknowledge
```

### Step 4: Test ML Backfill Integration

```bash
# Configure your environment for ML backfill
export INFLUXDB_URL=http://localhost:8086
export INFLUXDB_TOKEN=tradeknowledge-super-secret-token
export INFLUXDB_ORG=tradeknowledge
export INFLUXDB_BUCKET=market_data

# Test ML backfill connection
python start_ml_backfill.py
```

---

## 🏗️ Infrastructure Components

### Core Services

**InfluxDB (Time-Series Database):**
- **Port**: 8086
- **Version**: 2.7-alpine (latest stable)
- **Memory Allocation**: 1-4GB (depending on environment)
- **Persistent Storage**: Docker volumes with host backup

**Redis (Caching Layer):**
- **Port**: 6379
- **Memory Limit**: 256MB-512MB
- **Persistence**: AOF enabled for durability

**Qdrant (Vector Database):**
- **Ports**: 6333 (HTTP), 6334 (gRPC)
- **Purpose**: Embeddings and semantic search

### ML-Optimized InfluxDB Configuration

**Pre-Configured Buckets:**
```
market_data          # Primary market data (permanent retention)
crypto_data          # Cryptocurrency data (5 years)
crypto_orderbook     # Order book microstructure (1 year)
equity_data          # Equity market data (10 years)
equity_fundamentals  # Fundamental data (permanent)
futures_data         # Futures contracts (7 years)
options_data         # Options with Greeks (3 years)
ml_features          # Pre-computed ML features (3 years)
correlations         # Cross-asset correlations (5 years)
strategy_performance # Strategy PnL (permanent)
realtime_data        # Real-time tick data (30 days)
hft_signals          # HFT signals (90 days)
system_metrics       # System monitoring (1 year)
data_quality         # Quality validation (1 year)
```

**Automated Tasks:**
- **Downsampling**: 1min → 5min → 1hr → 1day
- **Data Quality Monitoring**: Freshness and completeness checks
- **Cleanup**: Automatic removal of expired data

**Security Features:**
- **API Tokens**: Service-specific access tokens
- **Authentication**: Admin and service accounts
- **Network Isolation**: Docker network security

---

## 🔐 Security Configuration

### API Tokens

The system creates specialized tokens for different services:

```bash
# View created tokens
cat data/influxdb-backups/api_tokens.env

# Available tokens:
# - ml-backfill-service: Full market data access
# - strategy-execution: Read market data, write performance
# - realtime-trading: Real-time data access
# - analytics-readonly: Read-only access to all data
# - monitoring-service: System monitoring
```

### Production Security Hardening

```bash
# Update docker-compose.prod.yml for production:
# 1. Strong passwords and tokens
# 2. SSL certificates
# 3. Network security
# 4. Resource limits
```

---

## 📊 Environment-Specific Configurations

### Development Environment
```bash
# Lightweight setup for development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Features:
# - Reduced memory allocation
# - Debug logging enabled
# - Hot reload for API
# - Development tokens
```

### Production Environment
```bash
# Full production stack
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Features:
# - SSL/TLS encryption
# - Resource limits and monitoring
# - Backup automation
# - Performance optimization
```

### Monitoring Environment
```bash
# Enable monitoring stack
docker-compose --profile monitoring up -d

# Includes:
# - Prometheus metrics collection
# - Grafana dashboards
# - System monitoring
```

---

## 🔍 Monitoring and Operations

### Health Checks

```bash
# Check all services
docker-compose ps

# InfluxDB health
curl -s http://localhost:8086/health | jq

# Service logs
docker-compose logs -f influxdb
docker-compose logs -f tradeknowledge-api
```

### Performance Monitoring

```bash
# InfluxDB metrics
curl -s http://localhost:8086/metrics

# Resource usage
docker stats

# Disk usage
docker system df
du -sh data/influxdb-backups/
```

### Backup and Recovery

```bash
# Manual backup
docker exec tradeknowledge-influxdb influx backup /backups/manual-backup-$(date +%Y%m%d)

# Automated backup (scheduled via cron)
0 2 * * * docker exec tradeknowledge-influxdb influx backup /backups/daily-backup-$(date +%Y%m%d)

# Restore from backup
docker exec tradeknowledge-influxdb influx restore /backups/backup-directory
```

---

## ⚡ Performance Optimization

### Memory Tuning

**Development (Low Resource):**
- InfluxDB: 512MB-1GB
- Redis: 256MB
- Total: ~1GB RAM

**Production (Optimized):**
- InfluxDB: 2-4GB
- Redis: 512MB-1GB
- Total: ~4-6GB RAM

### Storage Optimization

**Retention Policies:**
- **High-frequency data**: 30-90 days
- **Daily market data**: 5-10 years
- **Strategy performance**: Permanent
- **ML features**: 3 years

**Downsampling Strategy:**
- 1-minute → 5-minute (real-time)
- 5-minute → 1-hour (hourly)
- 1-hour → daily (daily)

### Query Performance

```sql
-- Optimized queries for ML feature extraction
from(bucket: "market_data")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "equity_prices_1min")
  |> filter(fn: (r) => r.symbol == "SPY")
  |> aggregateWindow(every: 5m, fn: last)
```

---

## 🛠️ Troubleshooting

### Common Issues

**InfluxDB Won't Start:**
```bash
# Check logs
docker-compose logs influxdb

# Common causes:
# - Insufficient memory
# - Port conflicts
# - Permission issues
# - Corrupted data directory
```

**Connection Refused:**
```bash
# Verify service is running
docker-compose ps influxdb

# Check network connectivity
docker exec tradeknowledge-api curl -s http://influxdb:8086/health

# Verify environment variables
docker exec tradeknowledge-api env | grep INFLUXDB
```

**Performance Issues:**
```bash
# Monitor resource usage
docker stats tradeknowledge-influxdb

# Check query performance
docker exec tradeknowledge-influxdb influx query 'from(bucket:"market_data") |> range(start:-1h) |> count()'

# Optimize retention policies
docker exec tradeknowledge-influxdb influx bucket list
```

### Recovery Procedures

**Data Corruption:**
```bash
# Stop services
docker-compose down

# Restore from backup
docker volume rm tradeknowledge_influxdb_data
docker-compose up -d influxdb

# Restore data
docker exec tradeknowledge-influxdb influx restore /backups/latest-backup
```

**Configuration Reset:**
```bash
# Reset InfluxDB configuration
docker-compose down
docker volume rm tradeknowledge_influxdb_config
docker-compose up -d influxdb
```

---

## 🎯 Next Steps

### Immediate Actions
1. **Configure Environment**: Set up .env with your API keys
2. **Start Services**: `docker-compose up -d`
3. **Verify Setup**: Check health endpoints and logs
4. **Test ML Backfill**: Run `python start_ml_backfill.py`

### Production Deployment
1. **Security Review**: Configure strong passwords and SSL
2. **Resource Planning**: Allocate appropriate memory and storage
3. **Monitoring Setup**: Enable Prometheus and Grafana
4. **Backup Strategy**: Configure automated backups

### Integration
1. **API Integration**: Update applications to use containerized InfluxDB
2. **Strategy Development**: Begin implementing arbitrage strategies
3. **ML Pipeline**: Set up feature engineering and model training
4. **Performance Monitoring**: Establish operational dashboards

---

## 📋 Quick Reference

### Useful Commands
```bash
# Start full stack
docker-compose up -d

# Start development stack
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Start production stack
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# View logs
docker-compose logs -f influxdb

# Execute InfluxDB commands
docker exec -it tradeknowledge-influxdb influx
```

### Important URLs
- **InfluxDB UI**: http://localhost:8086
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **API Documentation**: http://localhost:8000/docs

### Default Credentials
- **InfluxDB**: admin / tradeknowledge123
- **Grafana**: admin / admin
- **Token**: tradeknowledge-super-secret-token

---

*🐳 Your TradeKnowledge system now has enterprise-grade persistent storage with ML-optimized configuration for professional algorithmic trading!*