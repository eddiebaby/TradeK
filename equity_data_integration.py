#!/usr/bin/env python3
"""
Equity Data Integration for LDES System

Complete integration of IEX Cloud Free and Polygon.io data sources
with the existing LDES InfluxDB system for real-time equity data.
"""

import asyncio
import logging
from datetime import datetime, time as dt_time
from typing import List
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.collectors.equity_data_collector import EquityDataCollector
from src.collectors.verification_service import VerificationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/equity_data.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class EquityDataIntegration:
    """Main integration service for equity data collection and verification"""
    
    def __init__(self, symbols: List[str]):
        """
        Initialize equity data integration
        
        Args:
            symbols: List of equity symbols to monitor
        """
        self.symbols = symbols
        self.collector: EquityDataCollector = None
        self.verification_service: VerificationService = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        # Initialize collector for real-time data
        self.collector = EquityDataCollector(
            symbols=self.symbols,
            collection_interval=15  # Every 15 seconds
        )
        await self.collector.__aenter__()
        
        # Initialize verification service for daily checks
        self.verification_service = VerificationService(
            symbols=self.symbols,
            verification_time=dt_time(16, 30)  # 4:30 PM ET
        )
        await self.verification_service.__aenter__()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.collector:
            await self.collector.__aexit__(exc_type, exc_val, exc_tb)
        if self.verification_service:
            await self.verification_service.__aexit__(exc_type, exc_val, exc_tb)
    
    async def start_services(self):
        """Start both collection and verification services"""
        logger.info("Starting equity data integration services")
        
        # Start real-time data collection
        await self.collector.start()
        logger.info("✅ Real-time data collection started")
        
        # Start daily verification service
        await self.verification_service.start()
        logger.info("✅ Daily verification service started")
        
        logger.info("🚀 All equity data services are running!")
    
    async def stop_services(self):
        """Stop all services"""
        logger.info("Stopping equity data integration services")
        
        if self.collector:
            await self.collector.stop()
            logger.info("✅ Real-time data collection stopped")
        
        if self.verification_service:
            await self.verification_service.stop()
            logger.info("✅ Verification service stopped")
    
    def get_status(self) -> dict:
        """Get status of all services"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "symbols_monitored": len(self.symbols),
            "collector_running": self.collector.is_running if self.collector else False,
            "verification_running": self.verification_service.is_running if self.verification_service else False
        }
        
        # Add performance metrics if available
        if self.collector:
            performance = self.collector.get_performance_summary()
            status["collector_performance"] = performance
        
        return status

async def main():
    """Main function to run the equity data integration"""
    print("🚀 Starting TradeKnowledge Equity Data Integration")
    print("=" * 60)
    
    # Define symbols to monitor (can be configured)
    symbols = [
        # Major tech stocks
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        # Financial sector
        "JPM", "BAC", "WFC", "GS", "MS",
        # Market indicators
        "SPY", "QQQ", "IWM", "VTI", "VXX",
        # Other popular stocks
        "TSLA", "NVDA", "AMD", "NFLX", "CRM"
    ]
    
    print(f"📊 Monitoring {len(symbols)} equity symbols")
    print(f"🔗 Data Sources: IEX Cloud Free + Polygon.io EOD")
    print(f"💾 Storage: InfluxDB (LDES system)")
    print("=" * 60)
    
    try:
        async with EquityDataIntegration(symbols) as integration:
            
            # Start all services
            await integration.start_services()
            
            # Print initial status
            status = integration.get_status()
            print(f"\n📈 Integration Status:")
            print(f"   Symbols: {status['symbols_monitored']}")
            print(f"   Collector: {'🟢 Running' if status['collector_running'] else '🔴 Stopped'}")
            print(f"   Verification: {'🟢 Running' if status['verification_running'] else '🔴 Stopped'}")
            
            print(f"\n💡 Integration running... Press Ctrl+C to stop")
            print(f"   📊 Real-time collection every 15 seconds")
            print(f"   ✅ Daily verification at 4:30 PM ET")
            print(f"   📁 Logs: logs/equity_data.log")
            
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(300)  # Status update every 5 minutes
                    
                    status = integration.get_status()
                    performance = status.get('collector_performance', {})
                    
                    print(f"\n⏰ Status Update - {status['timestamp'][:19]}")
                    if performance:
                        print(f"   Success Rate: {performance.get('success_rate', 0):.1f}%")
                        print(f"   Avg Collection Time: {performance.get('avg_collection_time_seconds', 0):.2f}s")
                        print(f"   Recent Collections: {performance.get('collections_analyzed', 0)}")
            
            except KeyboardInterrupt:
                print(f"\n\n🛑 Shutdown requested by user")
            
            # Graceful shutdown
            await integration.stop_services()
            print(f"✅ All services stopped gracefully")
    
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        print(f"❌ Integration failed: {e}")
        return 1
    
    print(f"🎉 Equity Data Integration completed successfully!")
    return 0

def setup_api_keys():
    """Help user set up API keys"""
    print("🔑 API Key Setup Guide")
    print("=" * 40)
    
    # Check current keys
    iex_token = os.getenv('IEX_CLOUD_API_TOKEN')
    polygon_key = os.getenv('POLYGON_API_KEY')
    
    print(f"IEX Cloud Token: {'✅ Set' if iex_token else '❌ Missing'}")
    print(f"Polygon API Key: {'✅ Set' if polygon_key else '❌ Missing'}")
    
    if not iex_token or not polygon_key:
        print(f"\n📝 To set up API keys:")
        print(f"1. IEX Cloud (Free): https://iexcloud.io/")
        print(f"   - Sign up for free account")
        print(f"   - Get publishable token (starts with 'pk_')")
        print(f"   - Add to .env: IEX_CLOUD_API_TOKEN=your_token")
        
        print(f"\n2. Polygon.io (Free): https://polygon.io/")
        print(f"   - Sign up for free account (5 API calls/minute)")
        print(f"   - Get API key from dashboard")
        print(f"   - Add to .env: POLYGON_API_KEY=your_key")
        
        print(f"\n💡 Note: IEX Cloud free tier works without token for basic quotes")
        print(f"   Polygon.io requires API key for all endpoints")
        
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TradeKnowledge Equity Data Integration")
    parser.add_argument("--setup", action="store_true", help="Show API key setup guide")
    parser.add_argument("--test", action="store_true", help="Run quick test mode")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_api_keys()
        sys.exit(0)
    
    # Check API keys before starting
    if not setup_api_keys():
        print(f"\n❌ Please set up API keys before running integration")
        print(f"   Run: python equity_data_integration.py --setup")
        sys.exit(1)
    
    # Run the integration
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n🛑 Integration stopped by user")
        sys.exit(0)