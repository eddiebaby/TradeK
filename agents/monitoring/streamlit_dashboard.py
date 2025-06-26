#!/usr/bin/env python3
"""
Streamlit Dashboard for Agent Trio Workflows
Interactive dashboard for monitoring and controlling agent workflows
"""

import asyncio
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st

# Add agents directory to path
sys.path.append(str(Path(__file__).parent.parent))

from influx_blackboard import (
    InfluxBlackboard, write_task, read_tasks, update_status, 
    get_context, log_performance
)

# Configure Streamlit page
st.set_page_config(
    page_title="Agent Trio Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def run_async(func):
    """Helper to run async functions in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(func)

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_agent_data():
    """Get agent data with caching"""
    try:
        blackboard = InfluxBlackboard()
        agents_data = {}
        
        for agent_name in ['Researcher', 'Mastermind', 'Executor']:
            tasks = run_async(read_tasks(agent_name))
            context = run_async(get_context(agent_name))
            
            agents_data[agent_name] = {
                'tasks': tasks,
                'context': context,
                'total_tasks': len(tasks),
                'pending_tasks': len([t for t in tasks if t.get('status') != 'completed']),
                'completed_tasks': len([t for t in tasks if t.get('status') == 'completed'])
            }
        
        return agents_data, True
    except Exception as e:
        return {}, False

def main():
    # Header
    st.title("🤖 Agent Trio Monitoring Dashboard")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("🔧 Controls")
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    # Get data
    agents_data, connection_ok = get_agent_data()
    
    if not connection_ok:
        st.error("❌ Unable to connect to InfluxDB blackboard")
        st.stop()
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 InfluxDB Status", "Connected ✅")
    
    with col2:
        total_tasks = sum(data['total_tasks'] for data in agents_data.values())
        st.metric("📝 Total Tasks", total_tasks)
    
    with col3:
        total_pending = sum(data['pending_tasks'] for data in agents_data.values())
        st.metric("⏳ Pending Tasks", total_pending)
    
    with col4:
        total_completed = sum(data['completed_tasks'] for data in agents_data.values())
        st.metric("✅ Completed Tasks", total_completed)
    
    st.markdown("---")
    
    # Agent overview
    st.header("🎯 Agent Overview")
    
    col1, col2, col3 = st.columns(3)
    
    # Agent cards
    agents = [
        ("Researcher", "🔍", col1),
        ("Mastermind", "🧠", col2), 
        ("Executor", "⚡", col3)
    ]
    
    for agent_name, icon, col in agents:
        with col:
            data = agents_data.get(agent_name, {})
            context = data.get('context', {})
            
            if context and not context.get('error'):
                success_rate = context.get('success_rate', 0) * 100
                efficiency = context.get('efficiency_score', 0)
                avg_tokens = context.get('avg_tokens', 0)
            else:
                success_rate = efficiency = avg_tokens = 0
            
            st.subheader(f"{icon} {agent_name}")
            st.metric("Pending Tasks", data.get('pending_tasks', 0))
            st.metric("Success Rate", f"{success_rate:.1f}%")
            st.metric("Efficiency", f"{efficiency:.2f}")
            st.metric("Avg Tokens", f"{avg_tokens:.0f}")
    
    st.markdown("---")
    
    # Task creation
    st.header("➕ Create New Task")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_agent = st.selectbox("Agent", ["Researcher", "Mastermind", "Executor"])
    
    with col2:
        operation_map = {
            "Researcher": ["security_intelligence", "market_intelligence", "technical_analysis"],
            "Mastermind": ["architecture_design", "strategic_planning", "quality_strategy"],
            "Executor": ["implementation", "testing", "deployment"]
        }
        selected_operation = st.selectbox("Operation", operation_map[selected_agent])
    
    with col3:
        task_description = st.text_input("Task Description", "")
    
    if st.button("🚀 Create Task"):
        if task_description:
            try:
                task_id = run_async(write_task(selected_agent, selected_operation, task_description))
                st.success(f"✅ Task created: {task_id}")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error creating task: {e}")
        else:
            st.warning("Please enter a task description")
    
    st.markdown("---")
    
    # Visualizations
    st.header("📊 Analytics")
    
    # Task distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Task Distribution")
        
        task_data = []
        for agent_name, data in agents_data.items():
            task_data.append({
                'Agent': agent_name,
                'Pending': data.get('pending_tasks', 0),
                'Completed': data.get('completed_tasks', 0)
            })
        
        if task_data:
            df = pd.DataFrame(task_data)
            fig = px.bar(df, x='Agent', y=['Pending', 'Completed'], 
                        title="Task Status by Agent",
                        color_discrete_map={'Pending': '#ffc107', 'Completed': '#28a745'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Agent Performance")
        
        perf_data = []
        for agent_name, data in agents_data.items():
            context = data.get('context', {})
            if context and not context.get('error'):
                perf_data.append({
                    'Agent': agent_name,
                    'Success Rate': context.get('success_rate', 0) * 100,
                    'Efficiency': context.get('efficiency_score', 0) * 100
                })
        
        if perf_data:
            df = pd.DataFrame(perf_data)
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=df['Success Rate'],
                theta=df['Agent'],
                fill='toself',
                name='Success Rate'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=df['Efficiency'],
                theta=df['Agent'],
                fill='toself',
                name='Efficiency'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Agent Performance Radar"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent tasks table
    st.header("📋 Recent Tasks")
    
    all_tasks = []
    for agent_name, data in agents_data.items():
        tasks = data.get('tasks', [])
        for task in tasks[-10:]:  # Last 10 tasks per agent
            all_tasks.append({
                'Agent': agent_name,
                'Operation': task.get('operation', 'unknown'),
                'Status': task.get('status', 'pending'),
                'Task ID': task.get('task_id', 'unknown')[:8]  # Shortened ID
            })
    
    if all_tasks:
        df = pd.DataFrame(all_tasks)
        
        # Color-code status
        def color_status(val):
            colors = {
                'completed': 'background-color: #d4edda',
                'processing': 'background-color: #fff3cd', 
                'pending': 'background-color: #f8f9fa',
                'failed': 'background-color: #f8d7da'
            }
            return colors.get(val, '')
        
        styled_df = df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No recent tasks found")
    
    # Footer
    st.markdown("---")
    st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("💡 Dashboard auto-refreshes every 30 seconds when enabled")

if __name__ == "__main__":
    main()