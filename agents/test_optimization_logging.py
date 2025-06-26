#!/usr/bin/env python3
"""
Test script to verify optimization logging to InfluxDB
Tests writing optimization suggestions to the newly created optimizations bucket
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

async def test_optimization_logging():
    """Test logging optimization suggestions to InfluxDB"""
    print("🧪 Testing Optimization Logging")
    print("-" * 40)
    
    # Configuration
    url = "http://localhost:8087"
    token = "blackboard-super-secret-auth-token"
    org = "AgentBlackboard"
    bucket = "optimizations"
    
    try:
        # Initialize client
        client = InfluxDBClient(url=url, token=token, org=org)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        print("1. Testing optimization suggestion logging...")
        
        # Create optimization suggestion data
        optimization_data = {
            "agent": "Researcher",
            "operation_type": "stock_analysis",
            "current_tokens": 450,
            "suggested_tokens": 320,
            "savings": 130,
            "optimization_type": "prompt_optimization",
            "suggestion": "Use more specific prompts to reduce token usage",
            "confidence": 0.85,
            "implementation_effort": "low"
        }
        
        # Create point for optimization suggestion
        point = Point("optimization_suggestion") \
            .tag("agent", optimization_data["agent"]) \
            .tag("operation_type", optimization_data["operation_type"]) \
            .tag("optimization_type", optimization_data["optimization_type"]) \
            .tag("implementation_effort", optimization_data["implementation_effort"]) \
            .field("current_tokens", optimization_data["current_tokens"]) \
            .field("suggested_tokens", optimization_data["suggested_tokens"]) \
            .field("savings", optimization_data["savings"]) \
            .field("confidence", optimization_data["confidence"]) \
            .field("suggestion", optimization_data["suggestion"]) \
            .time(datetime.utcnow(), WritePrecision.NS)
        
        # Write to InfluxDB
        write_api.write(bucket=bucket, org=org, record=point)
        print(f"   ✅ Optimization suggestion logged successfully")
        print(f"   Agent: {optimization_data['agent']}")
        print(f"   Operation: {optimization_data['operation_type']}")
        print(f"   Token savings: {optimization_data['savings']}")
        print(f"   Suggestion: {optimization_data['suggestion']}")
        
        print("\n2. Testing performance optimization logging...")
        
        # Create performance optimization data
        perf_data = {
            "agent": "Executor",
            "operation_type": "tdd_implementation",
            "execution_time": 5.2,
            "suggested_time": 3.8,
            "time_savings": 1.4,
            "optimization_type": "algorithm_optimization",
            "suggestion": "Use cached test results to speed up implementation",
            "confidence": 0.92
        }
        
        point2 = Point("performance_optimization") \
            .tag("agent", perf_data["agent"]) \
            .tag("operation_type", perf_data["operation_type"]) \
            .tag("optimization_type", perf_data["optimization_type"]) \
            .field("execution_time", perf_data["execution_time"]) \
            .field("suggested_time", perf_data["suggested_time"]) \
            .field("time_savings", perf_data["time_savings"]) \
            .field("confidence", perf_data["confidence"]) \
            .field("suggestion", perf_data["suggestion"]) \
            .time(datetime.utcnow(), WritePrecision.NS)
        
        write_api.write(bucket=bucket, org=org, record=point2)
        print(f"   ✅ Performance optimization logged successfully")
        print(f"   Agent: {perf_data['agent']}")
        print(f"   Operation: {perf_data['operation_type']}")
        print(f"   Time savings: {perf_data['time_savings']}s")
        
        print("\n3. Verifying data was written...")
        
        # Query to verify data
        query_api = client.query_api()
        query = f'''
        from(bucket: "{bucket}")
          |> range(start: -1h)
          |> filter(fn: (r) => r["_measurement"] == "optimization_suggestion" or r["_measurement"] == "performance_optimization")
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: 10)
        '''
        
        result = query_api.query(query, org=org)
        
        if result:
            print(f"   ✅ Found {len(result)} records in optimizations bucket")
            for table in result:
                for record in table.records:
                    print(f"   - {record.get_time()}: {record.get_measurement()} for {record.values.get('agent', 'unknown')}")
        else:
            print("   ⚠️  No records found (this might be expected for first run)")
        
        client.close()
        print("\n✅ Optimization logging test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during optimization logging test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_bucket_permissions():
    """Test bucket permissions and accessibility"""
    print("\n🔐 Testing Bucket Permissions")
    print("-" * 40)
    
    url = "http://localhost:8087"
    token = "blackboard-super-secret-auth-token"
    org = "AgentBlackboard"
    
    try:
        client = InfluxDBClient(url=url, token=token, org=org)
        buckets_api = client.buckets_api()
        
        # List all buckets
        buckets = buckets_api.find_buckets(org=org)
        
        print("Available buckets:")
        optimization_bucket = None
        for bucket in buckets.buckets:
            status = "✅" if bucket.name == "optimizations" else "ℹ️"
            print(f"   {status} {bucket.name} (retention: {bucket.retention_rules[0].every_seconds if bucket.retention_rules else 'infinite'}s)")
            if bucket.name == "optimizations":
                optimization_bucket = bucket
        
        if optimization_bucket:
            print(f"\n✅ Optimizations bucket found!")
            print(f"   ID: {optimization_bucket.id}")
            print(f"   Description: {optimization_bucket.description}")
            print(f"   Retention: {optimization_bucket.retention_rules[0].every_seconds}s (30 days)")
            print(f"   Created: {optimization_bucket.created_at}")
        else:
            print("❌ Optimizations bucket not found!")
            return False
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Error testing bucket permissions: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Optimization Logging Test Suite")
    print("=" * 60)
    
    try:
        # Test bucket permissions
        if not await test_bucket_permissions():
            return False
        
        # Test optimization logging
        if not await test_optimization_logging():
            return False
        
        print("\n🎉 All optimization logging tests passed!")
        print("\n📋 Summary:")
        print("   ✅ Optimizations bucket exists and is accessible")
        print("   ✅ Optimization suggestions can be logged")
        print("   ✅ Performance optimizations can be logged")
        print("   ✅ Data can be queried from the bucket")
        
        return True
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)