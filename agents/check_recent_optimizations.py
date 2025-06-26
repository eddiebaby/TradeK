#!/usr/bin/env python3
"""
Check recent optimization entries to verify the system is working
"""

from influxdb_client import InfluxDBClient
from datetime import datetime

def check_recent_optimizations():
    """Check recent optimization suggestions"""
    print("🔍 Checking Recent Optimization Suggestions")
    print("-" * 50)
    
    # Connect to InfluxDB
    client = InfluxDBClient(
        url="http://localhost:8087",
        token="blackboard-super-secret-auth-token",
        org="AgentBlackboard"
    )
    
    query_api = client.query_api()
    
    # Query recent optimizations
    query = '''
    from(bucket: "optimizations")
      |> range(start: -2h)
      |> filter(fn: (r) => r["_measurement"] == "optimizations")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: 10)
    '''
    
    try:
        result = query_api.query(query, org="AgentBlackboard")
        
        if result:
            print("Recent optimization suggestions:")
            count = 0
            for table in result:
                for record in table.records:
                    count += 1
                    timestamp = record.get_time()
                    agent = record.values.get('target_agent', 'unknown')
                    category = record.values.get('category', 'unknown')
                    suggestion = record.values.get('suggestion', 'no suggestion')[:80]
                    confidence = record.values.get('confidence', 0)
                    
                    print(f"  {count}. {timestamp} - {agent}")
                    print(f"     Category: {category}")
                    print(f"     Confidence: {confidence:.2f}")
                    print(f"     Suggestion: {suggestion}...")
                    print()
            
            if count == 0:
                print("  No optimization suggestions found in the last 2 hours.")
                print("  This is normal if the system hasn't been under heavy load.")
            else:
                print(f"✅ Found {count} optimization suggestions - system is working!")
        else:
            print("  No data returned from query.")
            
    except Exception as e:
        print(f"❌ Error querying optimizations: {e}")
    
    # Check if bucket exists and is accessible
    print("\n🔍 Checking optimizations bucket accessibility...")
    
    try:
        buckets_api = client.buckets_api()
        buckets = buckets_api.find_buckets(org="AgentBlackboard")
        
        opt_bucket = None
        for bucket in buckets.buckets:
            if bucket.name == "optimizations":
                opt_bucket = bucket
                break
        
        if opt_bucket:
            print(f"✅ Optimizations bucket exists:")
            print(f"   ID: {opt_bucket.id}")
            print(f"   Retention: {opt_bucket.retention_rules[0].every_seconds}s")
            print(f"   Created: {opt_bucket.created_at}")
        else:
            print("❌ Optimizations bucket not found!")
            
    except Exception as e:
        print(f"❌ Error checking bucket: {e}")
    
    client.close()

if __name__ == "__main__":
    check_recent_optimizations()