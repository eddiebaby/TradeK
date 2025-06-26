#!/bin/bash
# InfluxDB Installation Script for Ubuntu 24.04
# Run this script to install InfluxDB for TradeKnowledge

set -e

echo "🚀 Installing InfluxDB for TradeKnowledge..."

# Add InfluxDB repository
echo "📦 Adding InfluxDB repository..."
wget -q https://repos.influxdata.com/influxdata-archive_compat.key
echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null

echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list

# Update package list
echo "🔄 Updating package list..."
sudo apt-get update

# Install InfluxDB
echo "📥 Installing InfluxDB..."
sudo apt-get install influxdb2 influxdb2-client -y

# Start and enable InfluxDB service
echo "🔧 Starting InfluxDB service..."
sudo systemctl start influxdb
sudo systemctl enable influxdb

# Check if service is running
echo "✅ Checking InfluxDB status..."
sudo systemctl status influxdb --no-pager

echo ""
echo "🎉 InfluxDB installation completed!"
echo ""
echo "📋 Next steps:"
echo "1. Open your browser and go to: http://localhost:8086"
echo "2. Complete the initial setup (create admin user, org, bucket)"
echo "3. Copy the generated token to your .env file"
echo ""
echo "🔧 Configuration:"
echo "   INFLUXDB_URL=http://localhost:8086"
echo "   INFLUXDB_TOKEN=your_token_here"
echo "   INFLUXDB_ORG=your_org_name"
echo "   INFLUXDB_BUCKET=market_data"
echo ""
echo "💡 You can also run the automated setup with:"
echo "   python scripts/setup_influxdb.py"