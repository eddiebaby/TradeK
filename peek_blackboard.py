#!/usr/bin/env python3
"""
Comprehensive blackboard inspector for TradeKnowledge agents
Checks both file-based blackboard and InfluxDB storage
"""

import json
import os
from pathlib import Path
from datetime import datetime
import subprocess
import sys

def check_influxdb():
    """Check InfluxDB for agent data"""
    print("🏪 InfluxDB Status:")
    print("-" * 30)
    
    # Check environment variables
    url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
    token = os.getenv('INFLUXDB_TOKEN')
    org = os.getenv('INFLUXDB_ORG', 'TradeKnowledge')
    bucket = os.getenv('INFLUXDB_BUCKET', 'data')
    
    print(f"URL: {url}")
    print(f"Org: {org}")
    print(f"Bucket: {bucket}")
    print(f"Token: {'✅ Set' if token else '❌ Not set'}")
    
    if not token:
        print("❌ InfluxDB token not configured")
        return
    
    try:
        # Try to query InfluxDB
        from influxdb_client import InfluxDBClient
        
        with InfluxDBClient(url=url, token=token, org=org) as client:
            # Check if we can connect
            ping = client.ping()
            print(f"Connection: {'✅ Connected' if ping else '❌ Failed'}")
            
            # Query for agent data
            query_api = client.query_api()
            
            # Look for agent measurements
            query = f'''
            from(bucket: "{bucket}")
            |> range(start: -24h)
            |> group(columns: ["_measurement"])
            |> distinct(column: "_measurement")
            '''
            
            tables = query_api.query(query)
            measurements = []
            for table in tables:
                for record in table.records:
                    measurements.append(record.get_value())
            
            if measurements:
                print(f"Measurements found: {measurements}")
                
                # Get recent agent data
                agent_query = f'''
                from(bucket: "{bucket}")
                |> range(start: -1h)
                |> filter(fn: (r) => r._measurement =~ /agent|blackboard|task/)
                |> limit(n: 10)
                '''
                
                tables = query_api.query(agent_query)
                if tables:
                    print("Recent agent data:")
                    for table in tables:
                        for record in table.records:
                            print(f"  {record.get_time()}: {record.get_measurement()} = {record.get_value()}")
            else:
                print("No measurements found in InfluxDB")
    
    except ImportError:
        print("❌ influxdb-client not installed. Install with: pip install influxdb-client")
    except Exception as e:
        print(f"❌ InfluxDB connection failed: {e}")
    
    print()

def peek_blackboard():
    """Inspect the current state of the agent blackboard"""
    agents_dir = Path(__file__).parent / "agents"
    
    print("🔍 TradeKnowledge Agent Blackboard Inspector")
    print("=" * 50)
    
    # Check for blackboard files
    blackboard_file = agents_dir / "blackboard.md"
    cache_file = agents_dir / "data_cache.json"
    
    print(f"📍 Checking directory: {agents_dir}")
    print(f"📝 Blackboard file: {blackboard_file}")
    print(f"💾 Cache file: {cache_file}")
    print()
    
    # Check blackboard.md
    if blackboard_file.exists():
        print("✅ Found blackboard.md:")
        print("-" * 30)
        with open(blackboard_file, 'r') as f:
            content = f.read()
            print(content[:1000] + ("..." if len(content) > 1000 else ""))
        print()
    else:
        print("❌ No blackboard.md found")
        print()
    
    # Check data_cache.json
    if cache_file.exists():
        print("✅ Found data_cache.json:")
        print("-" * 30)
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                print(f"Cache entries: {len(data.get('cache', {}))}")
                print(f"Last updated: {data.get('last_updated', 'Unknown')}")
                
                # Show cache keys
                cache_keys = list(data.get('cache', {}).keys())[:10]
                if cache_keys:
                    print(f"Recent cache keys: {cache_keys}")
        except Exception as e:
            print(f"Error reading cache: {e}")
        print()
    else:
        print("❌ No data_cache.json found")
        print()
    
    # Check for agent logs
    logs_dir = agents_dir / "logs"
    if logs_dir.exists():
        print("📋 Agent logs:")
        print("-" * 30)
        log_files = list(logs_dir.glob("*.log"))
        for log_file in log_files[:5]:  # Show first 5
            size = log_file.stat().st_size
            modified = datetime.fromtimestamp(log_file.stat().st_mtime)
            print(f"  {log_file.name}: {size} bytes, modified {modified}")
        print()
    else:
        print("❌ No logs directory found")
        print()
    
    # Check agent activity
    print("🤖 Agent Scripts:")
    print("-" * 30)
    agent_scripts = list(agents_dir.glob("ask_*.py"))
    for script in agent_scripts:
        print(f"  {script.name}")
    
    if not agent_scripts:
        print("  No agent scripts found")
    
    print()
    print("💡 To activate agents and create blackboard data:")
    print("   cd agents && python ask_researcher.py")
    print("   cd agents && python ask_mastermind.py") 
    print("   cd agents && python ask_executor.py")

if __name__ == "__main__":
    # Load environment variables
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    peek_blackboard()
    check_influxdb()