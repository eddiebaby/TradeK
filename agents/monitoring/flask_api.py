#!/usr/bin/env python3
"""
Flask API for Agent Trio Monitoring and Control
Provides REST endpoints for agent status, metrics, and control operations
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import yaml

# Add agents directory to path
sys.path.append(str(Path(__file__).parent.parent))

from influx_blackboard import (
    InfluxBlackboard, write_task, read_tasks, update_status, 
    get_context, log_performance
)

app = Flask(__name__)
CORS(app)

# Initialize blackboard
blackboard = InfluxBlackboard()

# HTML template for web interface
WEB_INTERFACE = """
<!DOCTYPE html>
<html>
<head>
    <title>Agent Trio Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .agent-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .agent-card { border-left: 4px solid #007acc; }
        .researcher { border-left-color: #28a745; }
        .mastermind { border-left-color: #ffc107; }
        .executor { border-left-color: #dc3545; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .metric-value { font-weight: bold; }
        .status-active { color: #28a745; }
        .status-processing { color: #ffc107; }
        .status-error { color: #dc3545; }
        .btn { padding: 8px 16px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; }
        .btn-primary { background: #007acc; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-warning { background: #ffc107; color: black; }
        h1 { color: #333; text-align: center; }
        h2 { color: #555; }
        .refresh-btn { position: fixed; top: 20px; right: 20px; }
    </style>
    <script>
        function refreshData() {
            location.reload();
        }
        
        function createTask(agent, operation, description) {
            fetch('/api/tasks', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({agent, operation, description})
            }).then(() => refreshData());
        }
        
        // Auto-refresh every 30 seconds
        setInterval(refreshData, 30000);
    </script>
</head>
<body>
    <div class="container">
        <h1>🤖 Agent Trio Monitoring Dashboard</h1>
        <button class="btn btn-primary refresh-btn" onclick="refreshData()">🔄 Refresh</button>
        
        <div class="card">
            <h2>📊 System Overview</h2>
            <div class="metric">
                <span>InfluxDB Status:</span>
                <span class="metric-value status-active">{{ influx_status }}</span>
            </div>
            <div class="metric">
                <span>Total Tasks:</span>
                <span class="metric-value">{{ total_tasks }}</span>
            </div>
            <div class="metric">
                <span>Last Update:</span>
                <span class="metric-value">{{ last_update }}</span>
            </div>
        </div>
        
        <div class="agent-grid">
            {% for agent in agents %}
            <div class="card agent-card {{ agent.name.lower() }}">
                <h2>{{ agent.icon }} {{ agent.name }}</h2>
                <div class="metric">
                    <span>Pending Tasks:</span>
                    <span class="metric-value">{{ agent.pending_tasks }}</span>
                </div>
                <div class="metric">
                    <span>Success Rate:</span>
                    <span class="metric-value">{{ "%.1f"|format(agent.success_rate * 100) }}%</span>
                </div>
                <div class="metric">
                    <span>Efficiency:</span>
                    <span class="metric-value">{{ "%.2f"|format(agent.efficiency) }}</span>
                </div>
                <div class="metric">
                    <span>Avg Tokens:</span>
                    <span class="metric-value">{{ "%.0f"|format(agent.avg_tokens) }}</span>
                </div>
                <button class="btn btn-success" 
                        onclick="createTask('{{ agent.name }}', 'analysis', 'Quick analysis task')">
                    ➕ Add Task
                </button>
            </div>
            {% endfor %}
        </div>
        
        <div class="card">
            <h2>📝 Recent Tasks</h2>
            <div style="max-height: 300px; overflow-y: auto;">
                {% for task in recent_tasks %}
                <div class="metric">
                    <span>{{ task.agent }} - {{ task.operation }}</span>
                    <span class="metric-value status-{{ task.status }}">{{ task.status }}</span>
                </div>
                {% endfor %}
            </div>
        </div>
    </div>
</body>
</html>
"""

def run_async(func):
    """Helper to run async functions in Flask routes"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(func)
    finally:
        loop.close()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    try:
        # Get agent data
        agents_data = []
        total_tasks = 0
        
        for agent_name in ['Researcher', 'Mastermind', 'Executor']:
            tasks = run_async(read_tasks(agent_name))
            context = run_async(get_context(agent_name))
            
            pending_tasks = len([t for t in tasks if t.get('status') != 'completed'])
            total_tasks += len(tasks)
            
            if context and not context.get('error'):
                success_rate = context.get('success_rate', 0)
                efficiency = context.get('efficiency_score', 0)
                avg_tokens = context.get('avg_tokens', 0)
            else:
                success_rate = efficiency = avg_tokens = 0
            
            icon = {'Researcher': '🔍', 'Mastermind': '🧠', 'Executor': '⚡'}[agent_name]
            
            agents_data.append({
                'name': agent_name,
                'icon': icon,
                'pending_tasks': pending_tasks,
                'success_rate': success_rate,
                'efficiency': efficiency,
                'avg_tokens': avg_tokens
            })
        
        # Get recent tasks
        recent_tasks = []
        for agent_name in ['Researcher', 'Mastermind', 'Executor']:
            tasks = run_async(read_tasks(agent_name))
            for task in tasks[-5:]:  # Last 5 tasks
                recent_tasks.append({
                    'agent': agent_name,
                    'operation': task.get('operation', 'unknown'),
                    'status': task.get('status', 'pending')
                })
        
        # Check InfluxDB status
        try:
            health = blackboard.client.health()
            influx_status = "Connected ✅"
        except:
            influx_status = "Disconnected ❌"
        
        return render_template_string(WEB_INTERFACE,
            agents=agents_data,
            total_tasks=total_tasks,
            recent_tasks=recent_tasks[-10:],  # Last 10 tasks
            influx_status=influx_status,
            last_update=datetime.now().strftime('%H:%M:%S')
        )
        
    except Exception as e:
        return f"Error loading dashboard: {e}", 500

@app.route('/api/status')
def api_status():
    """Get overall system status"""
    try:
        status = {
            'timestamp': datetime.now().isoformat(),
            'influxdb': blackboard.client.health().status,
            'agents': {}
        }
        
        for agent_name in ['Researcher', 'Mastermind', 'Executor']:
            tasks = run_async(read_tasks(agent_name))
            context = run_async(get_context(agent_name))
            
            status['agents'][agent_name.lower()] = {
                'total_tasks': len(tasks),
                'pending_tasks': len([t for t in tasks if t.get('status') != 'completed']),
                'context': context
            }
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agents/<agent_name>')
def api_agent_details(agent_name):
    """Get detailed agent information"""
    try:
        agent_name = agent_name.capitalize()
        tasks = run_async(read_tasks(agent_name))
        context = run_async(get_context(agent_name))
        
        return jsonify({
            'agent': agent_name,
            'tasks': tasks,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks', methods=['POST'])
def api_create_task():
    """Create a new task for an agent"""
    try:
        data = request.json
        agent = data['agent']
        operation = data['operation']
        description = data['description']
        
        task_id = run_async(write_task(agent, operation, description))
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'agent': agent,
            'operation': operation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks/<task_id>/status', methods=['PUT'])
def api_update_task_status(task_id):
    """Update task status"""
    try:
        data = request.json
        new_status = data['status']
        
        run_async(update_status(task_id, new_status))
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'status': new_status
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics')
def api_metrics():
    """Get agent performance metrics"""
    try:
        metrics = {}
        
        for agent_name in ['Researcher', 'Mastermind', 'Executor']:
            context = run_async(get_context(agent_name))
            if context and not context.get('error'):
                metrics[agent_name.lower()] = context
        
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Agent Trio Flask API")
    print("📊 Dashboard: http://localhost:5000")
    print("🔗 API: http://localhost:5000/api/status")
    app.run(host='0.0.0.0', port=5000, debug=True)