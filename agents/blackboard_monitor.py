#!/usr/bin/env python3
"""
Quick Blackboard Record Count Monitor
Simple utility to check blackboard activity and record counts

This provides a lightweight way to monitor the organic data accumulation
in the InfluxDB blackboard without running the full retention manager.
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
import sys

# Add agents directory to path
sys.path.append(str(Path(__file__).parent))

from influx_blackboard import generate_report, get_blackboard

async def get_quick_stats():
    """Get quick blackboard statistics"""
    try:
        blackboard = get_blackboard()
        
        if not blackboard.query_api:
            return {"error": "InfluxDB not available"}
        
        # Count records in different buckets
        buckets = ["tasks", "metrics", "reflections", "optimizations"]
        total_records = 0
        bucket_counts = {}
        
        for bucket in buckets:
            try:
                # Simple count query for each bucket
                query = f'''
                import "influxdata/influxdb/schema"
                
                from(bucket: "{bucket}")
                  |> range(start: -7d)
                  |> count()
                  |> sum()
                '''
                
                tables = blackboard.query_api.query(query)
                bucket_count = 0
                
                for table in tables:
                    for record in table.records:
                        value = record.get_value()
                        if value:
                            bucket_count += value
                
                bucket_counts[bucket] = bucket_count
                total_records += bucket_count
                
            except Exception as e:
                print(f"Warning: Could not query {bucket}: {e}")
                bucket_counts[bucket] = 0
        
        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_records_7d": total_records,
            "bucket_breakdown": bucket_counts,
            "blackboard_active": total_records > 0
        }
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}

async def check_blackboard_activity(hours_back=24):
    """Check blackboard activity for specified hours"""
    print(f"🔍 Checking blackboard activity (last {hours_back} hours)")
    print("=" * 50)
    
    stats = await get_quick_stats()
    
    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
        return
    
    print(f"📊 Blackboard Status:")
    print(f"   • Total Records (7d): {stats['total_records_7d']:,}")
    print(f"   • Blackboard Active: {'Yes' if stats['blackboard_active'] else 'No'}")
    
    if stats["bucket_breakdown"]:
        print(f"\n📋 Bucket Breakdown:")
        for bucket, count in stats["bucket_breakdown"].items():
            print(f"   • {bucket}: {count:,} records")
    
    # Estimate progress toward 1000 record goal
    estimated_records = stats['total_records_7d']
    target = 1000
    
    print(f"\n🎯 Progress Toward Target:")
    if estimated_records >= target:
        print(f"   ✅ Target achieved! {estimated_records:,} >= {target:,} records")
    else:
        remaining = target - estimated_records
        print(f"   ⏳ {remaining:,} more records needed to reach {target:,}")
        
        # Estimate time to target based on current rate
        if estimated_records > 0:
            daily_rate = estimated_records  # Records per day
            if daily_rate > 0:
                days_to_target = remaining / daily_rate
                print(f"   📈 At current rate: ~{days_to_target:.1f} days to reach target")

async def watch_blackboard(interval_minutes=10):
    """Watch blackboard activity continuously"""
    print(f"👁️  Watching blackboard activity (updates every {interval_minutes} minutes)")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        while True:
            await check_blackboard_activity()
            print(f"\n⏰ Next check in {interval_minutes} minutes...")
            await asyncio.sleep(interval_minutes * 60)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")

async def main():
    """Main function"""
    print("📊 BLACKBOARD ACTIVITY MONITOR")
    print("=" * 40)
    
    while True:
        print("\n📋 Commands:")
        print("1. 'check' - Check current activity")
        print("2. 'watch' - Continuous monitoring")
        print("3. 'quick' - Quick stats only")
        print("4. 'quit' - Exit")
        
        command = input("\n🔧 Enter command: ").strip().lower()
        
        if command in ['quit', 'exit', 'q']:
            break
            
        try:
            if command == 'check':
                await check_blackboard_activity()
                
            elif command == 'watch':
                interval = input("⏱️  Check interval in minutes (default: 10): ").strip()
                interval = int(interval) if interval else 10
                await watch_blackboard(interval)
                
            elif command == 'quick':
                stats = await get_quick_stats()
                if "error" in stats:
                    print(f"❌ Error: {stats['error']}")
                else:
                    print(f"📈 Records (7d): {stats['total_records_7d']:,}")
                    print(f"🔄 Blackboard Active: {'Yes' if stats['blackboard_active'] else 'No'}")
                    
            else:
                print("❌ Unknown command")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")
    except Exception as e:
        print(f"❌ Error: {e}")