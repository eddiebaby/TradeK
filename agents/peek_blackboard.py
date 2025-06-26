#!/usr/bin/env python3
"""
Simple Blackboard Data Explorer
Peek into InfluxDB to see what data was logged during our CAT stock analysis
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from influx_blackboard import get_blackboard

async def explore_blackboard():
    """Explore what's in the blackboard"""
    print("🔍 EXPLORING AGENT BLACKBOARD DATA")
    print("=" * 60)
    
    bb = get_blackboard()
    
    if not bb.query_api:
        print("❌ InfluxDB not available")
        return
    
    # Check what buckets exist
    print("\n📂 AVAILABLE BUCKETS:")
    try:
        buckets = bb.client.buckets_api().find_buckets()
        for bucket in buckets.buckets:
            print(f"   📁 {bucket.name}")
    except Exception as e:
        print(f"❌ Error listing buckets: {e}")
    
    # Check what measurements exist in key buckets
    buckets_to_check = ["tasks", "metrics", "blackboard", "data"]
    
    for bucket_name in buckets_to_check:
        print(f"\n📊 MEASUREMENTS IN '{bucket_name}' BUCKET:")
        
        query = f'''
        import "influxdata/influxdb/schema"
        schema.measurements(bucket: "{bucket_name}")
        '''
        
        try:
            tables = bb.query_api.query(query)
            measurements = []
            
            for table in tables:
                for record in table.records:
                    measurement = record.values.get('_value')
                    if measurement:
                        measurements.append(measurement)
            
            if measurements:
                for measurement in set(measurements):
                    print(f"   📈 {measurement}")
                    
                    # Get sample data from each measurement
                    sample_query = f'''
                    from(bucket: "{bucket_name}")
                      |> range(start: -24h)
                      |> filter(fn: (r) => r._measurement == "{measurement}")
                      |> limit(n: 3)
                    '''
                    
                    try:
                        sample_tables = bb.query_api.query(sample_query)
                        sample_count = 0
                        
                        for sample_table in sample_tables:
                            for sample_record in sample_table.records:
                                if sample_count == 0:
                                    print(f"      Sample fields: {list(sample_record.values.keys())}")
                                sample_count += 1
                        
                        if sample_count > 0:
                            print(f"      Records found: {sample_count}")
                        else:
                            print(f"      No recent records")
                    except Exception as e:
                        print(f"      Error getting sample: {e}")
            else:
                print(f"   (No measurements found)")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Check recent metrics specifically
    print(f"\n🎯 RECENT METRICS (Last 24 Hours):")
    
    metrics_query = '''
    from(bucket: "metrics")
      |> range(start: -24h)
      |> filter(fn: (r) => r._measurement == "metrics")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: 10)
    '''
    
    try:
        tables = bb.query_api.query(metrics_query)
        
        for table in tables:
            for record in table.records:
                timestamp = record.get_time().strftime('%H:%M:%S') if record.get_time() else 'Unknown'
                agent = record.values.get('agent', 'Unknown')
                operation = record.values.get('operation', 'Unknown')
                tokens = record.values.get('tokens_used', 0)
                exec_time = record.values.get('exec_time', 0)
                success = record.values.get('success', 'Unknown')
                
                print(f"   [{timestamp}] Agent:{agent} Op:{operation} Tokens:{tokens} Time:{exec_time:.2f}s Success:{success}")
                
    except Exception as e:
        print(f"   ❌ Error querying metrics: {e}")
    
    # Check recent tasks
    print(f"\n📝 RECENT TASKS (Last 24 Hours):")
    
    tasks_query = '''
    from(bucket: "tasks")
      |> range(start: -24h)
      |> filter(fn: (r) => r._measurement == "tasks")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: 5)
    '''
    
    try:
        tables = bb.query_api.query(tasks_query)
        
        for table in tables:
            for record in table.records:
                timestamp = record.get_time().strftime('%H:%M:%S') if record.get_time() else 'Unknown'
                agent = record.values.get('agent', 'Unknown')
                task_type = record.values.get('type', 'Unknown')
                status = record.values.get('status', 'Unknown')
                task_id = record.values.get('id', 'Unknown')
                
                print(f"   [{timestamp}] Agent:{agent} Type:{task_type} Status:{status} ID:{task_id}")
                
    except Exception as e:
        print(f"   ❌ Error querying tasks: {e}")
    
    # Summary stats
    print(f"\n📈 SUMMARY STATISTICS:")
    
    # Count total records in each bucket
    for bucket_name in ["tasks", "metrics"]:
        count_query = f'''
        from(bucket: "{bucket_name}")
          |> range(start: -24h)
          |> count()
        '''
        
        try:
            tables = bb.query_api.query(count_query)
            total_count = 0
            
            for table in tables:
                for record in table.records:
                    count = record.values.get('_value', 0)
                    total_count += count
            
            print(f"   {bucket_name}: {total_count} records in last 24h")
            
        except Exception as e:
            print(f"   {bucket_name}: Error counting - {e}")
    
    bb.close()

if __name__ == "__main__":
    asyncio.run(explore_blackboard())