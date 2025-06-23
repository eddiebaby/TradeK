#!/usr/bin/env python3
"""Quick test of IEX Cloud basic functionality"""

import asyncio
import sys
import os
sys.path.append('src')

from src.data_sources.iex_cloud_client import IEXCloudClient

async def test_iex():
    print("🧪 Testing IEX Cloud Basic Functionality")
    
    async with IEXCloudClient() as client:
        try:
            # Test connection
            is_connected = await client.validate_connection()
            print(f"Connection: {'✅' if is_connected else '❌'}")
            
            if is_connected:
                # Test quote
                quote = await client.get_quote('AAPL')
                print(f"✅ IEX Cloud Test Successful!")
                print(f"AAPL: ${quote.latest_price} ({quote.change_percent:.2%})")
                print(f"Volume: {quote.latest_volume:,}")
                print(f"Previous Close: ${quote.previous_close}")
                return True
            else:
                print("❌ Could not connect to IEX Cloud")
                return False
                
        except Exception as e:
            print(f"❌ IEX Cloud Test Failed: {e}")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_iex())
    if success:
        print("\n🎉 IEX Cloud is working! You can proceed with the integration.")
    else:
        print("\n💡 IEX Cloud requires API token for full functionality.")
        print("   But the integration can work with free tier limitations.")