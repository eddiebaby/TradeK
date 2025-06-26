#!/usr/bin/env python3
"""
Simple InfluxDB connection test
"""

import asyncio
import logging
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client import Point
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_influxdb():
    """Simple InfluxDB test"""
    
    url = "http://localhost:8086"
    token = "blackboard-super-secret-auth-token"
    org = "tradeknowledge"  
    bucket = "market_data"
    
    try:
        async with InfluxDBClientAsync(url=url, token=token, org=org) as client:
            logger.info("InfluxDB client created")
            
            # Try to get write API
            write_api = client.write_api()
            logger.info("Write API obtained")
            
            # Try a simple write
            point = Point("test").field("value", 1.0).time(datetime.utcnow())
            
            try:
                await write_api.write(bucket=bucket, record=point)
                logger.info("✅ InfluxDB write successful!")
            except Exception as write_error:
                logger.error(f"Write error: {write_error}")
                logger.info("This is expected if org/bucket don't exist yet")
                
    except Exception as e:
        logger.error(f"Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(test_influxdb())