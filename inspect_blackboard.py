#!/usr/bin/env python3
"""
InfluxDB Blackboard Inspector for Agent Communication

This utility allows you to peek into the InfluxDB instance that the agents
are using as their blackboard for inter-agent communication.

Usage:
    python inspect_blackboard.py                    # Show overview
    python inspect_blackboard.py --tasks            # Show current tasks
    python inspect_blackboard.py --data             # Show data entries
    python inspect_blackboard.py --metrics          # Show performance metrics
    python inspect_blackboard.py --agent RESEARCHER # Show specific agent data
    python inspect_blackboard.py --live             # Live monitoring mode
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import influxdb_client
from influxdb_client.client.query_api import QueryApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BlackboardInspector:
    """Inspector for the InfluxDB agent blackboard."""
    
    def __init__(self):
        """Initialize the blackboard inspector."""
        self.url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.token = os.getenv("INFLUXDB_TOKEN")
        self.org = os.getenv("INFLUXDB_ORG", "TradeKnowledge")
        self.bucket = os.getenv("INFLUXDB_BUCKET", "data")
        
        if not self.token:
            raise ValueError("INFLUXDB_TOKEN not found in environment variables")
        
        self.client = None
        self.query_api = None
        
        # Agent mappings
        self.agent_names = {
            "R": "🔍 RESEARCHER",
            "M": "🧠 MASTERMIND", 
            "E": "⚡ EXECUTOR"
        }
        
        # Status icons
        self.status_icons = {
            "new": "🆕",
            "proc": "⏳", 
            "done": "✅"
        }
        
        # Priority icons
        self.priority_icons = {
            1: "🔴",
            2: "🟡", 
            3: "🟢"
        }
    
    def connect(self):
        """Connect to InfluxDB."""
        try:
            self.client = influxdb_client.InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org
            )
            self.query_api = self.client.query_api()
            
            # Test connection
            health = self.client.health()
            if health.status != "pass":
                raise RuntimeError(f"InfluxDB health check failed: {health.message}")
            
            print(f"✅ Connected to InfluxDB at {self.url}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to InfluxDB: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from InfluxDB."""
        if self.client:
            self.client.close()
            print("📡 Disconnected from InfluxDB")
    
    def query_measurements(self) -> List[str]:
        """Get list of available measurements (tables)."""
        try:
            query = f'''
                import "influxdata/influxdb/schema"
                schema.measurements(bucket: "{self.bucket}")
            '''
            
            result = self.query_api.query(query)
            measurements = []
            
            for table in result:
                for record in table.records:
                    measurements.append(record.get_value())
            
            return measurements
            
        except Exception as e:
            print(f"❌ Error querying measurements: {e}")
            return []
    
    def query_tasks(self, agent: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
        """Query agent tasks from the blackboard."""
        try:
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: -24h)
                |> filter(fn: (r) => r._measurement == "tasks")
            '''
            
            if agent:
                query += f'|> filter(fn: (r) => r.agent == "{agent.upper()}")'
            
            if status:
                query += f'|> filter(fn: (r) => r.status == "{status}")'
            
            query += '''
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"], desc: true)
            '''
            
            result = self.query_api.query(query)
            tasks = []
            
            for table in result:
                for record in table.records:
                    task = {
                        "time": record.get_time(),
                        "agent": record.values.get("agent", ""),
                        "status": record.values.get("status", ""),
                        "priority": record.values.get("priority", 3),
                        "type": record.values.get("type", ""),
                        "desc": record.values.get("desc", ""),
                        "deps": record.values.get("deps", "")
                    }
                    tasks.append(task)
            
            return tasks
            
        except Exception as e:
            print(f"❌ Error querying tasks: {e}")
            return []
    
    def query_data_entries(self, key_filter: Optional[str] = None) -> List[Dict]:
        """Query data entries from the blackboard."""
        try:
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: -24h)
                |> filter(fn: (r) => r._measurement == "data")
            '''
            
            if key_filter:
                query += f'|> filter(fn: (r) => r.k =~ /{key_filter}/)'
            
            query += '''
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"], desc: true)
                |> limit(n: 50)
            '''
            
            result = self.query_api.query(query)
            data_entries = []
            
            for table in result:
                for record in table.records:
                    entry = {
                        "time": record.get_time(),
                        "key": record.values.get("k", ""),
                        "source": record.values.get("src", ""),
                        "format": record.values.get("fmt", ""),
                        "value": record.values.get("v", ""),
                        "checksum": record.values.get("cs", "")
                    }
                    data_entries.append(entry)
            
            return data_entries
            
        except Exception as e:
            print(f"❌ Error querying data entries: {e}")
            return []
    
    def query_metrics(self, agent: Optional[str] = None) -> List[Dict]:
        """Query performance metrics."""
        try:
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: -24h)
                |> filter(fn: (r) => r._measurement == "metrics")
            '''
            
            if agent:
                query += f'|> filter(fn: (r) => r.agent == "{agent.upper()}")'
            
            query += '''
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"], desc: true)
                |> limit(n: 100)
            '''
            
            result = self.query_api.query(query)
            metrics = []
            
            for table in result:
                for record in table.records:
                    metric = {
                        "time": record.get_time(),
                        "agent": record.values.get("agent", ""),
                        "operation": record.values.get("operation", ""),
                        "success": record.values.get("success", "true") == "true",
                        "tokens_used": record.values.get("tokens_used", 0),
                        "exec_time": record.values.get("exec_time", 0.0),
                        "accuracy": record.values.get("accuracy", 0.0)
                    }
                    metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error querying metrics: {e}")
            return []
    
    def query_reflections(self, agent: Optional[str] = None) -> List[Dict]:
        """Query agent reflections."""
        try:
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: -24h)
                |> filter(fn: (r) => r._measurement == "reflections")
            '''
            
            if agent:
                query += f'|> filter(fn: (r) => r.ag == "{agent.upper()}")'
            
            query += '''
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"], desc: true)
                |> limit(n: 50)
            '''
            
            result = self.query_api.query(query)
            reflections = []
            
            for table in result:
                for record in table.records:
                    reflection = {
                        "time": record.get_time(),
                        "agent": record.values.get("ag", ""),
                        "category": record.values.get("cat", ""),
                        "severity": record.values.get("sev", ""),
                        "note": record.values.get("n", ""),
                        "action": record.values.get("act", ""),
                        "impact": record.values.get("imp", 0.0)
                    }
                    reflections.append(reflection)
            
            return reflections
            
        except Exception as e:
            print(f"❌ Error querying reflections: {e}")
            return []
    
    def show_overview(self):
        """Show blackboard overview."""
        print("\n" + "="*60)
        print("🗂️  AGENT BLACKBOARD OVERVIEW")
        print("="*60)
        
        measurements = self.query_measurements()
        print(f"📊 Available Measurements: {', '.join(measurements) if measurements else 'None'}")
        
        # Get counts for each measurement
        for measurement in measurements:
            try:
                query = f'''
                    from(bucket: "{self.bucket}")
                    |> range(start: -24h)
                    |> filter(fn: (r) => r._measurement == "{measurement}")
                    |> count()
                '''
                
                result = self.query_api.query(query)
                count = 0
                for table in result:
                    for record in table.records:
                        count += record.get_value() or 0
                
                print(f"   {measurement}: {count} records (last 24h)")
                
            except Exception as e:
                print(f"   {measurement}: Error counting - {e}")
        
        # Show active tasks by agent
        print(f"\n📋 Active Tasks by Agent:")
        for agent_code, agent_name in self.agent_names.items():
            active_tasks = self.query_tasks(agent_code, "new") + self.query_tasks(agent_code, "proc")
            print(f"   {agent_name}: {len(active_tasks)} active")
        
        # Show recent activity
        recent_metrics = self.query_metrics()
        if recent_metrics:
            print(f"\n⚡ Recent Activity: {len(recent_metrics)} operations (last 24h)")
    
    def show_tasks(self, agent: Optional[str] = None, status: Optional[str] = None):
        """Show current tasks."""
        print("\n" + "="*60)
        print("📋 CURRENT TASKS")
        print("="*60)
        
        tasks = self.query_tasks(agent, status)
        
        if not tasks:
            print("📭 No tasks found")
            return
        
        # Group by agent
        by_agent = {}
        for task in tasks:
            agent_code = task["agent"]
            if agent_code not in by_agent:
                by_agent[agent_code] = []
            by_agent[agent_code].append(task)
        
        for agent_code, agent_tasks in by_agent.items():
            agent_name = self.agent_names.get(agent_code, agent_code)
            print(f"\n{agent_name}")
            print("-" * 40)
            
            for task in agent_tasks:
                status_icon = self.status_icons.get(task["status"], "❓")
                priority_icon = self.priority_icons.get(task["priority"], "⚪")
                time_str = task["time"].strftime("%H:%M:%S") if task["time"] else "Unknown"
                
                print(f"{status_icon} {priority_icon} {task['type']} - {time_str}")
                if task["desc"]:
                    print(f"   Description: {task['desc'][:80]}...")
                if task["deps"]:
                    print(f"   Dependencies: {task['deps']}")
                print()
    
    def show_data_entries(self, key_filter: Optional[str] = None):
        """Show data entries."""
        print("\n" + "="*60)
        print("💾 DATA ENTRIES")
        print("="*60)
        
        data_entries = self.query_data_entries(key_filter)
        
        if not data_entries:
            print("📭 No data entries found")
            return
        
        for entry in data_entries:
            time_str = entry["time"].strftime("%H:%M:%S") if entry["time"] else "Unknown"
            print(f"🔑 {entry['key']} - {time_str}")
            print(f"   Source: {entry['source']}")
            print(f"   Format: {entry['format']}")
            
            # Show value preview
            value = str(entry["value"])
            if len(value) > 100:
                print(f"   Value: {value[:100]}...")
            else:
                print(f"   Value: {value}")
            
            if entry["checksum"]:
                print(f"   Checksum: {entry['checksum'][:16]}...")
            print()
    
    def show_metrics(self, agent: Optional[str] = None):
        """Show performance metrics."""
        print("\n" + "="*60)
        print("📊 PERFORMANCE METRICS")
        print("="*60)
        
        metrics = self.query_metrics(agent)
        
        if not metrics:
            print("📭 No metrics found")
            return
        
        # Summary stats
        total_tokens = sum(int(m["tokens_used"]) for m in metrics)
        avg_exec_time = sum(float(m["exec_time"]) for m in metrics) / len(metrics)
        success_rate = sum(1 for m in metrics if m["success"]) / len(metrics) * 100
        
        print(f"📈 Summary (last 24h):")
        print(f"   Total operations: {len(metrics)}")
        print(f"   Total tokens used: {total_tokens:,}")
        print(f"   Average execution time: {avg_exec_time:.2f}s")
        print(f"   Success rate: {success_rate:.1f}%")
        
        # Recent metrics
        print(f"\n🕐 Recent Operations:")
        for metric in metrics[:20]:  # Show last 20
            agent_name = self.agent_names.get(metric["agent"], metric["agent"])
            time_str = metric["time"].strftime("%H:%M:%S") if metric["time"] else "Unknown"
            success_icon = "✅" if metric["success"] else "❌"
            
            print(f"{success_icon} {agent_name} {metric['operation']} - {time_str}")
            print(f"   Tokens: {metric['tokens_used']}, Time: {metric['exec_time']:.2f}s")
            print()
    
    def show_reflections(self, agent: Optional[str] = None):
        """Show agent reflections."""
        print("\n" + "="*60)
        print("🤔 AGENT REFLECTIONS")
        print("="*60)
        
        reflections = self.query_reflections(agent)
        
        if not reflections:
            print("📭 No reflections found")
            return
        
        for reflection in reflections:
            agent_name = self.agent_names.get(reflection["agent"], reflection["agent"])
            time_str = reflection["time"].strftime("%H:%M:%S") if reflection["time"] else "Unknown"
            
            print(f"💭 {agent_name} - {reflection['category']} - {time_str}")
            print(f"   Severity: {reflection['severity']}")
            print(f"   Note: {reflection['note']}")
            if reflection["action"]:
                print(f"   Action: {reflection['action']}")
            print(f"   Impact Score: {reflection['impact']}")
            print()
    
    def live_monitor(self, interval: int = 5):
        """Live monitoring mode."""
        print("\n" + "="*60)
        print("📡 LIVE BLACKBOARD MONITORING")
        print("="*60)
        print(f"Refreshing every {interval} seconds. Press Ctrl+C to stop.")
        
        try:
            while True:
                os.system('clear')  # Clear screen
                print(f"🕐 Live Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                self.show_overview()
                
                # Show recent tasks
                print(f"\n📋 Recent Tasks (last 10):")
                tasks = self.query_tasks()[:10]
                for task in tasks:
                    agent_name = self.agent_names.get(task["agent"], task["agent"])
                    status_icon = self.status_icons.get(task["status"], "❓")
                    time_str = task["time"].strftime("%H:%M:%S") if task["time"] else "Unknown"
                    print(f"   {status_icon} {agent_name} {task['type']} - {time_str}")
                
                print(f"\n💤 Waiting {interval}s for next update...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Inspect Agent Blackboard in InfluxDB")
    
    # Mode selection
    parser.add_argument("--tasks", action="store_true", help="Show current tasks")
    parser.add_argument("--data", action="store_true", help="Show data entries")
    parser.add_argument("--metrics", action="store_true", help="Show performance metrics")
    parser.add_argument("--reflections", action="store_true", help="Show agent reflections")
    parser.add_argument("--live", action="store_true", help="Live monitoring mode")
    
    # Filters
    parser.add_argument("--agent", choices=["RESEARCHER", "MASTERMIND", "EXECUTOR"], 
                       help="Filter by specific agent")
    parser.add_argument("--status", choices=["new", "proc", "done"], 
                       help="Filter tasks by status")
    parser.add_argument("--key-filter", help="Filter data entries by key pattern")
    parser.add_argument("--interval", type=int, default=5, 
                       help="Live monitoring refresh interval (seconds)")
    
    args = parser.parse_args()
    
    # Create inspector
    inspector = BlackboardInspector()
    
    try:
        # Connect to InfluxDB
        if not inspector.connect():
            return 1
        
        # Execute based on arguments
        if args.live:
            inspector.live_monitor(args.interval)
        elif args.tasks:
            inspector.show_tasks(args.agent, args.status)
        elif args.data:
            inspector.show_data_entries(args.key_filter)
        elif args.metrics:
            inspector.show_metrics(args.agent)
        elif args.reflections:
            inspector.show_reflections(args.agent)
        else:
            # Default: show overview
            inspector.show_overview()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    finally:
        inspector.disconnect()


if __name__ == "__main__":
    exit(main())