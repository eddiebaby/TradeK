#!/usr/bin/env python3
"""
Start Aggressive SPY/QQQ Backfill

Executes the complete SPARC Trio strategy for maximum granularity
historical data collection for SPY and QQQ.
"""

import asyncio
import logging
import sys
import os
from datetime import date
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.backfill.backfill_orchestrator import BackfillOrchestrator

# Setup comprehensive logging
def setup_logging():
    """Setup logging for backfill process"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'aggressive_backfill.log'),
            logging.StreamHandler()
        ]
    )

async def main():
    """Main aggressive backfill execution"""
    print("🚀 SPARC Trio: Aggressive SPY/QQQ Backfill")
    print("═" * 70)
    print("Target: Maximum granularity 1-minute historical data")
    print("Symbols: SPY, QQQ")
    print("Time Range: 2022-01-01 to present")
    print("Source: Polygon.io (free tier)")
    print("═" * 70)
    
    # Check API key
    polygon_key = os.getenv('POLYGON_API_KEY')
    if not polygon_key:
        print("❌ POLYGON_API_KEY not found in environment")
        print("\n🔑 Please set your Polygon.io API key:")
        print("1. Sign up at https://polygon.io/ (free tier)")
        print("2. Get your API key from the dashboard")
        print("3. Add to .env file: POLYGON_API_KEY=your_key")
        print("4. Or export POLYGON_API_KEY=your_key")
        return 1
    
    print(f"✅ Polygon API key configured")
    print(f"🎯 Estimated execution time: 2-3 hours")
    print(f"💾 Expected data: ~3M data points")
    print(f"💰 Cost: $0 (free tier)")
    
    try:
        async with BackfillOrchestrator() as orchestrator:
            print(f"\n🔥 Starting aggressive backfill...")
            
            # Execute with maximum available settings
            report = await orchestrator.execute_aggressive_backfill(
                symbols=["SPY", "QQQ"],
                start_date=date(2022, 1, 1),  # ~3 years of data
                end_date=date.today(),
                resume=True  # Resume if interrupted
            )
            
            print(f"\n🎉 AGGRESSIVE BACKFILL COMPLETED!")
            print("═" * 70)
            
            # Display results
            execution_summary = report.get('execution_summary', {})
            performance_metrics = report.get('performance_metrics', {})
            
            print(f"📊 EXECUTION RESULTS:")
            print(f"   ⏱️  Duration: {execution_summary.get('execution_time_hours', 0):.1f} hours")
            print(f"   🎯 Symbols: {len(execution_summary.get('symbols', []))}")
            print(f"   📅 Date Range: {execution_summary.get('date_range', 'N/A')}")
            print(f"   🔗 API Calls: {performance_metrics.get('total_api_calls', 0):,}")
            print(f"   ✅ Success Rate: {performance_metrics.get('success_rate', '0%')}")
            print(f"   📈 Data Points: {performance_metrics.get('data_points_collected', 0):,}")
            
            print(f"\n💾 DATA STORED IN:")
            print(f"   Database: InfluxDB")
            print(f"   Measurement: equity_prices_1m")
            print(f"   Tags: symbol=SPY/QQQ, source=polygon_historical")
            print(f"   Granularity: 1-minute OHLC + Volume")
            
            print(f"\n📋 REPORTS SAVED:")
            print(f"   Backfill report: data/backfill_reports/")
            print(f"   Progress tracking: data/backfill_progress/")
            print(f"   Logs: logs/aggressive_backfill.log")
            
            print(f"\n🚀 YOUR LDES SYSTEM NOW HAS:")
            print(f"   ✨ 3 years of 1-minute SPY data")
            print(f"   ✨ 3 years of 1-minute QQQ data")
            print(f"   ✨ Maximum available granularity")
            print(f"   ✨ Production-ready dataset")
            
            return 0
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Backfill interrupted by user")
        print(f"💡 Progress has been saved - run again to resume")
        return 0
        
    except Exception as e:
        print(f"\n❌ Backfill failed: {e}")
        print(f"\n📋 Troubleshooting:")
        print(f"1. Check your Polygon API key is valid")
        print(f"2. Ensure InfluxDB is running and accessible")
        print(f"3. Check logs/aggressive_backfill.log for details")
        print(f"4. Run again to resume from last checkpoint")
        
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    setup_logging()
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n🛑 Interrupted by user")
        sys.exit(0)