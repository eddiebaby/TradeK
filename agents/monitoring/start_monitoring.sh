#!/bin/bash
# Agent Trio Monitoring Stack Startup Script

echo "🚀 Starting Agent Trio Monitoring Stack"
echo "========================================"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Start Grafana and ensure InfluxDB is running
echo "🐳 Starting Grafana container..."
docker-compose up -d

# Wait for Grafana to be ready
echo "⏳ Waiting for Grafana to start..."
sleep 10

# Check if containers are running
if docker ps | grep -q "agent-grafana"; then
    echo "✅ Grafana is running"
else
    echo "❌ Grafana failed to start"
    exit 1
fi

if docker ps | grep -q "influxdb-blackboard"; then
    echo "✅ InfluxDB blackboard is running"
else
    echo "❌ InfluxDB blackboard is not running"
    exit 1
fi

echo ""
echo "🎉 Monitoring Stack Started Successfully!"
echo "========================================"
echo "📊 Grafana Dashboard: http://localhost:3000"
echo "   Username: admin"
echo "   Password: agentmonitoring123"
echo ""
echo "🌐 Flask API: http://localhost:5000"
echo "📈 Streamlit Dashboard: http://localhost:8501"
echo "🗄️  InfluxDB UI: http://localhost:8087"
echo ""
echo "Next steps:"
echo "1. Start Flask API: python flask_api.py"
echo "2. Start Streamlit: streamlit run streamlit_dashboard.py"
echo "3. Open Grafana and explore the Agent Trio dashboard"