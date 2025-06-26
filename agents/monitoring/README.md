# Agent Trio Web Monitoring Stack

Complete web monitoring solution for the Agent Trio system with InfluxDB blackboard.

## 🎯 Components

### 1. **Grafana Dashboard** (Port 3000)
- **Real-time metrics** from InfluxDB
- **Agent performance tracking** 
- **Task processing timelines**
- **Success rate monitoring**
- **Token usage analytics**

### 2. **Flask API** (Port 5000)
- **RESTful API** for agent control
- **Web dashboard** with live updates
- **Task creation** and status management
- **Agent metrics** and system status

### 3. **Streamlit Dashboard** (Port 8501)
- **Interactive workflows** and analytics
- **Visual task management**
- **Performance charts** and graphs
- **Real-time agent monitoring**

## 🚀 Quick Start

```bash
cd /home/scottschweizer/TradeKnowledge/agents/monitoring

# Start the monitoring stack
./start_monitoring.sh

# In separate terminals:
# Start Flask API
python flask_api.py

# Start Streamlit dashboard  
streamlit run streamlit_dashboard.py
```

## 📊 Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana** | http://localhost:3000 | admin / agentmonitoring123 |
| **Flask API** | http://localhost:5000 | No auth required |
| **Streamlit** | http://localhost:8501 | No auth required |
| **InfluxDB** | http://localhost:8087 | admin / blackboard123 |

## 🔧 Features

### Grafana Dashboard
- ✅ Agent task overview statistics
- ✅ Success rate pie charts
- ✅ Task processing timelines
- ✅ Performance metrics tables
- ✅ Token usage tracking
- ✅ Efficiency score gauges

### Flask API Endpoints
- `GET /` - Web dashboard
- `GET /api/status` - System status
- `GET /api/agents/<name>` - Agent details
- `POST /api/tasks` - Create new task
- `PUT /api/tasks/<id>/status` - Update task status
- `GET /api/metrics` - Performance metrics

### Streamlit Features
- ✅ Interactive agent overview
- ✅ Task creation interface
- ✅ Real-time charts and graphs
- ✅ Performance radar charts
- ✅ Recent tasks table
- ✅ Auto-refresh capability

## 🛠️ Configuration

### Grafana Data Source
- **Type**: InfluxDB
- **URL**: http://influxdb-blackboard:8086
- **Organization**: AgentBlackboard
- **Token**: blackboard-super-secret-auth-token

### InfluxDB Buckets
- `tasks` - Agent task data (7d retention)
- `metrics` - Performance metrics (30d retention)
- `blackboard` - General communication (no expiry)
- `reflections` - Agent insights (90d retention)
- `patterns` - Learned patterns (365d retention)

## 📈 Monitoring Capabilities

### Real-time Tracking
- ✅ Task creation and completion
- ✅ Agent performance metrics
- ✅ Success rate monitoring
- ✅ Token usage analytics
- ✅ Efficiency scoring

### Historical Analysis
- ✅ Performance trends over time
- ✅ Task processing patterns
- ✅ Agent workload distribution
- ✅ Success rate evolution

### Alerting (Future)
- Performance degradation alerts
- High token usage warnings
- Task failure notifications
- System health monitoring

## 🔄 Data Flow

```
Agent Trio → InfluxDB Blackboard → Grafana/Flask/Streamlit
     ↓              ↓                        ↓
  Task Data    Time Series Data        Visual Dashboards
  Metrics      Performance Stats       Interactive Controls
  Status       Historical Trends       Real-time Updates
```

## 🎉 Benefits

1. **Comprehensive Monitoring**: Full visibility into agent operations
2. **Multiple Interfaces**: Choose the right tool for your needs
3. **Real-time Updates**: Live data with automatic refresh
4. **Interactive Control**: Create tasks and manage workflows
5. **Historical Analysis**: Track trends and patterns over time
6. **Professional Dashboards**: Production-ready monitoring setup

Your agent trio now has enterprise-grade monitoring capabilities! 🚀