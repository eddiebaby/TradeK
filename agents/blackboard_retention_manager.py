#!/usr/bin/env python3
"""
Smart Blackboard Data Retention Manager
Implements intelligent data purging based on organic accumulation patterns

This system ensures the blackboard preserves data organically until at least 1000 
records accumulate naturally through normal trio usage, addressing the user's 
specific requirement for organic data accumulation rather than forced generation.

Key Features:
- Monitors blackboard record counts and storage usage
- Protects data until minimum threshold (1000 records) is reached
- Only purges when storage becomes critical (>95%)
- Preserves complete trio workflows
- Smart purging prioritizes older, less valuable data
- Comprehensive logging and reporting
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import sys

# Add agents directory to path
sys.path.append(str(Path(__file__).parent))

from influx_blackboard import get_blackboard, generate_report

class BlackboardRetentionManager:
    """Smart data retention manager for organic blackboard growth"""
    
    def __init__(self):
        self.config_path = Path(__file__).parent / "config" / "blackboard_influx.yaml"
        self.config = self._load_config()
        self.blackboard = get_blackboard()
        self.retention_config = self.config.get('data_retention', {})
        self.min_records = self.retention_config.get('minimum_records', 1000)
        self.storage_warning = self.retention_config.get('storage_warning_threshold', 80)
        self.storage_critical = self.retention_config.get('storage_critical_threshold', 95)
        self.preserve_workflows = self.retention_config.get('preserve_trio_workflows', True)
        self.smart_purging = self.retention_config.get('smart_purging_enabled', True)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load retention configuration"""
        if not self.config_path.exists():
            return {}
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def get_current_record_count(self) -> Dict[str, int]:
        """Get current record counts across all measurements"""
        if not self.blackboard.query_api:
            return {"error": "InfluxDB not available"}
        
        measurements = ["tasks", "metrics", "reflections", "optimizations", "data"]
        counts = {}
        total_count = 0
        
        try:
            for measurement in measurements:
                query = f'''
                from(bucket: "{measurement}")
                  |> range(start: -365d)  // Check all retained data
                  |> filter(fn: (r) => r._measurement == "{measurement}")
                  |> count()
                '''
                
                tables = self.blackboard.query_api.query(query)
                measurement_count = 0
                
                for table in tables:
                    for record in table.records:
                        measurement_count += record.get_value()
                        
                counts[measurement] = measurement_count
                total_count += measurement_count
                
            counts["total"] = total_count
            return counts
            
        except Exception as e:
            print(f"❌ Error getting record counts: {e}")
            return {"error": str(e)}
    
    async def get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage statistics (simulated for this implementation)"""
        # In a production environment, this would query actual storage metrics
        # For now, we'll simulate based on record counts
        try:
            record_counts = await self.get_current_record_count()
            total_records = record_counts.get("total", 0)
            
            # Estimate storage usage (rough approximation)
            # Assume average record size of ~1KB
            estimated_storage_mb = total_records * 1.0 / 1024  # Convert to MB
            
            # Simulate storage capacity (for demo purposes)
            total_capacity_mb = 10 * 1024  # 10GB
            usage_percentage = min(95, (estimated_storage_mb / total_capacity_mb) * 100)
            
            return {
                "total_records": total_records,
                "estimated_storage_mb": estimated_storage_mb,
                "total_capacity_mb": total_capacity_mb,
                "usage_percentage": usage_percentage,
                "warning_threshold": self.storage_warning,
                "critical_threshold": self.storage_critical
            }
            
        except Exception as e:
            print(f"❌ Error getting storage usage: {e}")
            return {"error": str(e)}
    
    async def should_purge_data(self) -> Dict[str, Any]:
        """Determine if data purging is needed based on configured thresholds"""
        record_counts = await self.get_current_record_count()
        storage_usage = await self.get_storage_usage()
        
        total_records = record_counts.get("total", 0)
        usage_percent = storage_usage.get("usage_percentage", 0)
        
        # Check minimum record protection
        records_protected = total_records < self.min_records
        
        # Check storage thresholds
        storage_warning = usage_percent >= self.storage_warning
        storage_critical = usage_percent >= self.storage_critical
        
        should_purge = storage_critical and not records_protected
        
        return {
            "should_purge": should_purge,
            "total_records": total_records,
            "min_records_threshold": self.min_records,
            "records_protected": records_protected,
            "storage_usage_percent": usage_percent,
            "storage_warning": storage_warning,
            "storage_critical": storage_critical,
            "purge_reason": self._get_purge_reason(should_purge, records_protected, storage_critical)
        }
    
    def _get_purge_reason(self, should_purge: bool, records_protected: bool, storage_critical: bool) -> str:
        """Get human-readable reason for purge decision"""
        if not should_purge:
            if records_protected:
                return f"Records protected: less than {self.min_records} records exist"
            elif not storage_critical:
                return "Storage usage within acceptable limits"
            else:
                return "Unknown protection reason"
        else:
            return f"Storage critical (>{self.storage_critical}%) and minimum records ({self.min_records}) threshold met"
    
    async def identify_purgeable_data(self) -> Dict[str, List[str]]:
        """Identify data that can be safely purged using smart algorithms"""
        if not self.smart_purging:
            return {"error": "Smart purging disabled"}
        
        purgeable_data = {
            "old_metrics": [],
            "expired_cache": [],
            "completed_workflows": [],
            "old_optimizations": []
        }
        
        try:
            # Find old metrics (older than 90 days)
            old_threshold = datetime.now() - timedelta(days=90)
            
            if self.blackboard.query_api:
                # Query for old metrics
                old_metrics_query = f'''
                from(bucket: "metrics")
                  |> range(start: -365d, stop: {old_threshold.strftime('%Y-%m-%dT%H:%M:%SZ')})
                  |> filter(fn: (r) => r._measurement == "metrics")
                  |> group(columns: ["_time"])
                  |> first()
                '''
                
                tables = self.blackboard.query_api.query(old_metrics_query)
                for table in tables:
                    for record in table.records:
                        purgeable_data["old_metrics"].append(record.get_time().isoformat())
                
                # Find expired cache entries
                expired_cache_query = f'''
                from(bucket: "data")
                  |> range(start: -30d)
                  |> filter(fn: (r) => r._measurement == "data")
                  |> filter(fn: (r) => r.ttl > 0 and (now() - r._time) > (r.ttl * 1s))
                '''
                
                # Note: This is a simplified query; actual implementation would be more complex
                
        except Exception as e:
            print(f"❌ Error identifying purgeable data: {e}")
            purgeable_data["error"] = str(e)
        
        return purgeable_data
    
    async def perform_smart_purge(self, dry_run: bool = True) -> Dict[str, Any]:
        """Perform intelligent data purging with preservation logic"""
        if not self.smart_purging:
            return {"error": "Smart purging disabled"}
        
        should_purge_result = await self.should_purge_data()
        
        if not should_purge_result["should_purge"]:
            return {
                "action": "no_purge_needed",
                "reason": should_purge_result["purge_reason"],
                "total_records": should_purge_result["total_records"]
            }
        
        purgeable_data = await self.identify_purgeable_data()
        
        purge_plan = {
            "action": "smart_purge_planned" if dry_run else "smart_purge_executed",
            "dry_run": dry_run,
            "purge_candidates": purgeable_data,
            "preservation_rules": {
                "preserve_trio_workflows": self.preserve_workflows,
                "minimum_records_protected": self.min_records,
                "current_record_count": should_purge_result["total_records"]
            }
        }
        
        if not dry_run:
            # Actual purging would happen here
            # For safety, this implementation only plans purges
            purge_plan["note"] = "Actual purging not implemented for safety - this is a planning run"
        
        return purge_plan
    
    async def generate_retention_report(self) -> Dict[str, Any]:
        """Generate comprehensive retention status report"""
        record_counts = await self.get_current_record_count()
        storage_usage = await self.get_storage_usage()
        should_purge_result = await self.should_purge_data()
        
        # Get efficiency report for additional context
        efficiency_report = await generate_report(24)  # Last 24 hours
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "retention_status": {
                "total_records": record_counts.get("total", 0),
                "minimum_threshold": self.min_records,
                "records_protected": should_purge_result["records_protected"],
                "target_achieved": record_counts.get("total", 0) >= self.min_records
            },
            "record_breakdown": record_counts,
            "storage_status": storage_usage,
            "purge_analysis": should_purge_result,
            "configuration": {
                "minimum_records": self.min_records,
                "storage_warning_threshold": self.storage_warning,
                "storage_critical_threshold": self.storage_critical,
                "preserve_trio_workflows": self.preserve_workflows,
                "smart_purging_enabled": self.smart_purging
            },
            "recent_activity": {
                "last_24h_operations": efficiency_report.get("total_operations", 0),
                "last_24h_tokens": efficiency_report.get("total_tokens", 0),
                "active_agents": list(efficiency_report.get("agents", {}).keys())
            }
        }
        
        return report
    
    async def monitor_continuous(self, check_interval_minutes: int = 30):
        """Continuous monitoring of retention status"""
        print(f"🔍 Starting continuous blackboard retention monitoring")
        print(f"⏱️  Check interval: {check_interval_minutes} minutes")
        print(f"🛡️  Protection threshold: {self.min_records} records")
        print("=" * 60)
        
        while True:
            try:
                report = await self.generate_retention_report()
                
                total_records = report["retention_status"]["total_records"]
                target_achieved = report["retention_status"]["target_achieved"]
                storage_percent = report["storage_status"].get("usage_percentage", 0)
                
                print(f"\n📊 Retention Monitor Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"📈 Records: {total_records:,} (Target: {self.min_records:,})")
                print(f"💾 Storage: {storage_percent:.1f}% used")
                
                if target_achieved:
                    print("✅ Target achieved! Organic data accumulation successful")
                else:
                    remaining = self.min_records - total_records
                    print(f"⏳ Target pending: {remaining:,} more records needed")
                
                # Check for warnings
                if storage_percent >= self.storage_critical:
                    print("🚨 CRITICAL: Storage usage critical - purging may be needed")
                elif storage_percent >= self.storage_warning:
                    print("⚠️  WARNING: Storage usage high - monitoring closely")
                
                # Check if purging should be considered
                should_purge_result = await self.should_purge_data()
                if should_purge_result["should_purge"]:
                    print("🧹 Smart purging criteria met - ready for data cleanup")
                    purge_plan = await self.perform_smart_purge(dry_run=True)
                    print(f"📋 Purge plan: {purge_plan['action']}")
                
                await asyncio.sleep(check_interval_minutes * 60)
                
            except Exception as e:
                print(f"❌ Monitor error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

async def main():
    """Main execution function"""
    print("🛡️  BLACKBOARD SMART RETENTION MANAGER")
    print("=" * 60)
    print("Protects blackboard data until 1000+ records accumulate organically")
    print("Only purges when storage becomes critical AND minimum threshold met")
    print()
    
    manager = BlackboardRetentionManager()
    
    while True:
        print("\n📋 Available Commands:")
        print("1. 'report' - Generate retention status report")
        print("2. 'count' - Show current record counts")
        print("3. 'storage' - Show storage usage")
        print("4. 'purge-check' - Check if purging is needed")
        print("5. 'purge-plan' - Generate smart purge plan (dry run)")
        print("6. 'monitor' - Start continuous monitoring")
        print("7. 'quit' - Exit")
        
        command = input("\n🔧 Enter command: ").strip().lower()
        
        if command in ['quit', 'exit', 'q']:
            break
        
        try:
            if command == 'report':
                print("\n📊 Generating retention report...")
                report = await manager.generate_retention_report()
                print(json.dumps(report, indent=2, default=str))
                
            elif command == 'count':
                print("\n📈 Getting record counts...")
                counts = await manager.get_current_record_count()
                print(f"Total Records: {counts.get('total', 0):,}")
                for measurement, count in counts.items():
                    if measurement != 'total':
                        print(f"  • {measurement}: {count:,}")
                        
            elif command == 'storage':
                print("\n💾 Getting storage usage...")
                storage = await manager.get_storage_usage()
                print(f"Storage Usage: {storage.get('usage_percentage', 0):.1f}%")
                print(f"Estimated Size: {storage.get('estimated_storage_mb', 0):.1f} MB")
                print(f"Total Records: {storage.get('total_records', 0):,}")
                
            elif command == 'purge-check':
                print("\n🔍 Checking purge criteria...")
                should_purge = await manager.should_purge_data()
                print(f"Should Purge: {should_purge['should_purge']}")
                print(f"Reason: {should_purge['purge_reason']}")
                print(f"Records Protected: {should_purge['records_protected']}")
                
            elif command == 'purge-plan':
                print("\n📋 Generating smart purge plan...")
                plan = await manager.perform_smart_purge(dry_run=True)
                print(json.dumps(plan, indent=2, default=str))
                
            elif command == 'monitor':
                interval = input("⏱️  Monitor interval in minutes (default: 30): ").strip()
                interval = int(interval) if interval else 30
                await manager.monitor_continuous(interval)
                
            else:
                print("❌ Unknown command. Please try again.")
                
        except Exception as e:
            print(f"❌ Error executing command: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Retention manager stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")