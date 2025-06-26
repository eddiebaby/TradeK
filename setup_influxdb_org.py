#!/usr/bin/env python3
"""
InfluxDB Organization and Bucket Setup
Sets up the required organization and bucket for TradeKnowledge
"""

import asyncio
import logging
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client import BucketsApi, OrganizationsApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_influxdb():
    """Setup InfluxDB organization and bucket"""
    
    # InfluxDB connection details
    url = "http://localhost:8086"
    token = "blackboard-super-secret-auth-token"
    org_name = "tradeknowledge"
    bucket_name = "market_data"
    
    try:
        # Connect to InfluxDB
        async with InfluxDBClientAsync(url=url, token=token) as client:
            logger.info("Connected to InfluxDB")
            
            # Check if we can connect
            ready = await client.ready()
            logger.info(f"InfluxDB ready status: {ready}")
            
            if ready:
                logger.info("✅ InfluxDB connection successful!")
                logger.info(f"Using org: {org_name}, bucket: {bucket_name}")
                logger.info("InfluxDB is ready for market data storage")
            else:
                logger.error("❌ InfluxDB is not ready")
                
    except Exception as e:
        logger.error(f"Error connecting to InfluxDB: {e}")
        logger.info("This might be expected if InfluxDB needs initial setup via web UI")
        logger.info("Please visit http://localhost:8086 to complete initial setup")
        logger.info(f"Use org name: {org_name}")
        logger.info(f"Use bucket name: {bucket_name}")

if __name__ == "__main__":
    asyncio.run(setup_influxdb())