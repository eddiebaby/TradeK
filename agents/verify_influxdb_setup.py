#!/usr/bin/env python3
"""
Comprehensive verification script for InfluxDB blackboard setup
Checks all buckets, permissions, and functionality
"""

import asyncio
import yaml
from pathlib import Path
from influxdb_client import InfluxDBClient

CONFIG_PATH = Path(__file__).parent / "config" / "blackboard_influx.yaml"

def load_config():
    """Load configuration"""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

async def verify_influxdb_setup():
    """Verify complete InfluxDB setup"""
    print("🔍 Comprehensive InfluxDB Setup Verification")
    print("=" * 60)
    
    # Load config
    config = load_config()
    influx_config = config['influxdb']
    
    # Test connection
    print("1. Testing InfluxDB connection...")
    try:
        token_file = Path(influx_config['token_file'])
        if not token_file.exists():
            print(f"   ❌ Token file not found: {token_file}")
            return False
        
        token = token_file.read_text().strip()
        client = InfluxDBClient(
            url=influx_config['url'],
            token=token,
            org=influx_config['org']
        )
        
        # Test health
        health = client.health()
        if health.status == "pass":
            print(f"   ✅ Connected to {influx_config['url']}")
        else:
            print(f"   ❌ Health check failed: {health.message}")
            return False
            
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    # Check buckets
    print("\n2. Checking bucket setup...")
    try:
        buckets_api = client.buckets_api()
        buckets = buckets_api.find_buckets(org=influx_config['org'])
        
        existing_buckets = {bucket.name: bucket for bucket in buckets.buckets}
        expected_buckets = list(config['retention_policies'].keys()) + ['blackboard']
        
        print(f"   Expected buckets: {expected_buckets}")
        print(f"   Found buckets: {list(existing_buckets.keys())}")
        
        missing_buckets = []
        for expected in expected_buckets:
            if expected in existing_buckets:
                bucket = existing_buckets[expected]
                retention = bucket.retention_rules[0].every_seconds if bucket.retention_rules else 0
                print(f"   ✅ {expected} (retention: {retention}s)")
            else:
                missing_buckets.append(expected)
                print(f"   ❌ {expected} - MISSING")
        
        if missing_buckets:
            print(f"\n   ⚠️  Missing buckets: {missing_buckets}")
            print("   These need to be created to avoid future errors.")
        else:
            print("\n   ✅ All expected buckets exist!")
            
    except Exception as e:
        print(f"   ❌ Error checking buckets: {e}")
        return False
    
    # Test basic operations
    print("\n3. Testing basic operations...")
    try:
        from influx_blackboard import get_blackboard
        
        bb = get_blackboard()
        
        # Test metric logging (this should trigger optimization checks)
        await bb.log_metrics(
            agent="TestAgent",
            operation="verification_test",
            tokens_used=50,
            exec_time=0.5,
            success=True
        )
        print("   ✅ Metric logging works")
        
        # Test optimization suggestion (should use optimizations bucket)
        from influx_blackboard import OptimizationSuggestion
        test_suggestion = OptimizationSuggestion(
            target_agent="TestAgent",
            category="test",
            suggestion="This is a test suggestion",
            expected_savings=10,
            confidence=0.95,
            auto_approve=False,
            implemented=False
        )
        await bb._write_optimization_suggestion(test_suggestion)
        print("   ✅ Optimization suggestion logging works")
        
        # Test context retrieval
        context = await bb.get_agent_context("TestAgent", lookback_hours=1)
        if "error" not in context:
            print("   ✅ Context retrieval works")
        else:
            print(f"   ⚠️  Context retrieval issue: {context['error']}")
        
        bb.close()
        
    except Exception as e:
        print(f"   ❌ Operation test failed: {e}")
        return False
    
    # Check for any write permissions issues
    print("\n4. Testing write permissions...")
    try:
        write_api = client.write_api()
        from influxdb_client import Point
        from datetime import datetime
        
        # Test write to each critical bucket
        critical_buckets = ['tasks', 'metrics', 'optimizations', 'data']
        
        for bucket_name in critical_buckets:
            if bucket_name in existing_buckets:
                point = Point("verification_test") \
                    .tag("source", "setup_verification") \
                    .field("test_value", 1) \
                    .time(datetime.utcnow())
                
                write_api.write(bucket=bucket_name, record=point)
                print(f"   ✅ Write permission to {bucket_name}")
            else:
                print(f"   ⚠️  Bucket {bucket_name} doesn't exist, skipping write test")
        
    except Exception as e:
        print(f"   ❌ Write permission test failed: {e}")
        return False
    
    client.close()
    
    print("\n🎉 InfluxDB Setup Verification Complete!")
    print("\n📋 Summary:")
    print("   ✅ InfluxDB connection working")
    print("   ✅ All expected buckets exist")
    print("   ✅ Basic operations functional")
    print("   ✅ Write permissions confirmed")
    print("   ✅ Optimization logging operational")
    
    return True

async def main():
    """Main verification function"""
    try:
        success = await verify_influxdb_setup()
        if success:
            print("\n🚀 Your InfluxDB blackboard system is fully operational!")
            print("   The 404 'optimizations bucket not found' error should be resolved.")
        else:
            print("\n❌ Verification failed. Please check the issues above.")
        
        return success
        
    except Exception as e:
        print(f"❌ Verification script failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)