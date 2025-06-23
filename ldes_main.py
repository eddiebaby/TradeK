#!/usr/bin/env python3
"""
LDES Main Orchestrator
=====================

Main entry point for the Liquidation Detection and Execution System.
Orchestrates data collection from multiple sources including Schwab API.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import signal
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ldes.core.config import LDESConfig, MarketDataConfig, InfluxDBConfig
from src.ldes.data.market_data_collector import MarketDataCollector
from src.ldes.data.schwab_client import create_schwab_provider
from src.ldes.data.influxdb_storage import create_influxdb_storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ldes.log')
    ]
)

logger = logging.getLogger(__name__)


class LDESOrchestrator:
    """
    Main orchestrator for the LDES system.
    
    Coordinates data collection, storage, and signal processing.
    """
    
    def __init__(self):
        """Initialize the LDES orchestrator."""
        self.config = LDESConfig()
        self.collector: Optional[MarketDataCollector] = None
        self.storage = None
        self.running = False
        
        # Symbols to track
        self.symbols = [
            # Major ETFs
            "SPY", "QQQ", "IWM", "DIA", "VTI", "VXUS",
            # Tech giants
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
            # Financial sector
            "JPM", "BAC", "WFC", "C", "GS", "MS",
            # Other major stocks
            "BRK.B", "JNJ", "V", "PG", "UNH", "HD",
            # Volatility products
            "VIX", "UVXY", "SVXY"
        ]
        
        logger.info("LDES Orchestrator initialized")
    
    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing LDES system...")
        
        try:
            # Initialize storage
            await self._setup_storage()
            
            # Initialize data collector
            await self._setup_data_collection()
            
            logger.info("LDES system initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize LDES system: {e}")
            raise
    
    async def _setup_storage(self) -> None:
        """Set up InfluxDB storage."""
        logger.info("Setting up InfluxDB storage...")
        
        # Check if InfluxDB is configured
        if not self.config.influxdb.url or not self.config.influxdb.token:
            logger.warning("InfluxDB not fully configured, using mock storage")
            self.storage = create_influxdb_storage(self.config.influxdb, use_mock=True)
        else:
            self.storage = create_influxdb_storage(self.config.influxdb, use_mock=False)
        
        # Connect to storage
        await self.storage.connect()
        logger.info(f"Storage connected: {self.storage.get_storage_info()}")
    
    async def _setup_data_collection(self) -> None:
        """Set up market data collection."""
        logger.info("Setting up market data collection...")
        
        # Create collector
        self.collector = MarketDataCollector(self.config, self.storage)
        
        # Add Schwab provider
        schwab_provider = create_schwab_provider(
            self.config.market_data, 
            use_mock=True  # Start with mock for testing
        )
        self.collector.add_provider("schwab", schwab_provider)
        
        logger.info("Market data collection setup complete")
    
    async def start_data_collection(self) -> None:
        """Start real-time data collection."""
        if not self.collector:
            raise RuntimeError("Data collector not initialized")
        
        logger.info("Starting data collection...")
        
        try:
            # Connect to all providers
            await self.collector.connect_all()
            
            # Subscribe to symbols
            await self.collector.subscribe_symbols(self.symbols)
            
            # Start collection
            await self.collector.start_collection()
            
            self.running = True
            logger.info(f"Data collection started for {len(self.symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to start data collection: {e}")
            raise
    
    async def stop_data_collection(self) -> None:
        """Stop data collection and cleanup."""
        if not self.running or not self.collector:
            return
        
        logger.info("Stopping data collection...")
        
        try:
            # Stop collection
            await self.collector.stop_collection()
            
            # Unsubscribe from symbols
            await self.collector.unsubscribe_symbols(self.symbols)
            
            # Disconnect from providers
            await self.collector.disconnect_all()
            
            self.running = False
            logger.info("Data collection stopped")
            
        except Exception as e:
            logger.error(f"Error stopping data collection: {e}")
    
    async def run_backfill(self, days_back: int = 7) -> None:
        """Run historical data backfill."""
        if not self.collector:
            raise RuntimeError("Data collector not initialized")
        
        logger.info(f"Starting historical backfill for {days_back} days...")
        
        try:
            await self.collector.backfill_historical_data(
                symbols=self.symbols[:5],  # Limit to first 5 symbols for testing
                days_back=days_back,
                timeframe="1min"
            )
            
            logger.info("Historical backfill completed")
            
        except Exception as e:
            logger.error(f"Backfill failed: {e}")
            raise
    
    async def run_monitoring_loop(self) -> None:
        """Run monitoring and status reporting loop."""
        logger.info("Starting monitoring loop...")
        
        while self.running:
            try:
                # Get collector status
                if self.collector:
                    status = self.collector.get_status()
                    logger.info(f"Data Collection Status: {status}")
                
                # Get storage info
                if self.storage:
                    storage_info = self.storage.get_storage_info()
                    logger.info(f"Storage Status: {storage_info}")
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief delay before retry
    
    async def shutdown(self) -> None:
        """Graceful shutdown of the system."""
        logger.info("Shutting down LDES system...")
        
        try:
            # Stop data collection
            await self.stop_data_collection()
            
            # Disconnect storage
            if self.storage:
                await self.storage.disconnect()
            
            logger.info("LDES system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def main():
    """Main entry point."""
    orchestrator = LDESOrchestrator()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler():
        logger.info("Received shutdown signal")
        orchestrator.running = False
    
    # Register signal handlers
    loop = asyncio.get_event_loop()
    for sig in [signal.SIGTERM, signal.SIGINT]:
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        # Initialize system
        await orchestrator.initialize()
        
        # Determine what to run based on command line arguments
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == "backfill":
                # Run historical backfill
                days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 7
                await orchestrator.run_backfill(days_back)
                return
            
            elif command == "test":
                # Run a quick test
                logger.info("Running test mode...")
                await orchestrator.start_data_collection()
                await asyncio.sleep(60)  # Collect for 1 minute
                await orchestrator.stop_data_collection()
                return
            
            elif command == "status":
                # Check system status
                logger.info("Checking system status...")
                if orchestrator.storage:
                    storage_info = orchestrator.storage.get_storage_info()
                    print(f"Storage: {storage_info}")
                return
        
        # Default: run full data collection
        logger.info("Starting full data collection mode...")
        
        # Start data collection
        await orchestrator.start_data_collection()
        
        # Run monitoring loop
        await orchestrator.run_monitoring_loop()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        raise
    finally:
        # Graceful shutdown
        await orchestrator.shutdown()


if __name__ == "__main__":
    # Ensure we have the proper environment
    if not os.path.exists(".env"):
        logger.error("No .env file found. Please create one with required configuration.")
        sys.exit(1)
    
    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)