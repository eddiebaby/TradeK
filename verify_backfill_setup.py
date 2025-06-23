#!/usr/bin/env python3
"""
Verify Backfill Setup

Quick verification that all components are ready for aggressive backfill.
"""

import os
import sys
from pathlib import Path
import requests
import asyncio

def check_api_key():
    """Check if Polygon API key is configured"""
    print("🔑 Checking API Key Setup...")
    
    polygon_key = os.getenv('POLYGON_API_KEY')
    if not polygon_key:
        print("❌ POLYGON_API_KEY not found in environment")
        print("💡 Add your Polygon.io API key to .env file:")
        print("   POLYGON_API_KEY=your_polygon_api_key")
        print("   Get free key at: https://polygon.io/")
        return False
    elif len(polygon_key.strip()) == 0:
        print("❌ POLYGON_API_KEY is empty")
        print("💡 Add your Polygon.io API key to .env file")
        return False
    else:
        print(f"✅ Polygon API key configured ({polygon_key[:8]}...)")
        return True

def check_influxdb():
    """Check if InfluxDB is running"""
    print("\n💾 Checking InfluxDB Connection...")
    
    influxdb_url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
    
    try:
        response = requests.get(f"{influxdb_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ InfluxDB is running at {influxdb_url}")
            return True
        else:
            print(f"❌ InfluxDB responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to InfluxDB at {influxdb_url}")
        print(f"   Error: {e}")
        print("💡 Make sure InfluxDB is running")
        return False

def check_directories():
    """Check/create required directories"""
    print("\n📁 Checking Directory Structure...")
    
    directories = [
        "logs",
        "data/backfill_progress", 
        "data/backfill_reports"
    ]
    
    all_good = True
    for dir_path in directories:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path} exists")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created {dir_path}")
            except Exception as e:
                print(f"❌ Failed to create {dir_path}: {e}")
                all_good = False
    
    return all_good

def check_dependencies():
    """Check required Python packages"""
    print("\n📦 Checking Dependencies...")
    
    required_packages = [
        "aiohttp",
        "influxdb-client", 
        "requests"
    ]
    
    all_good = True
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            print(f"   Install with: pip install {package}")
            all_good = False
    
    return all_good

async def test_polygon_api():
    """Test Polygon API connectivity"""
    print("\n🧪 Testing Polygon API...")
    
    polygon_key = os.getenv('POLYGON_API_KEY')
    if not polygon_key or len(polygon_key.strip()) == 0:
        print("❌ Cannot test - no API key configured")
        return False
    
    try:
        # Test with a simple API call
        url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev"
        params = {'apikey': polygon_key}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'OK':
                print("✅ Polygon API is working")
                return True
            else:
                print(f"❌ Polygon API error: {data}")
                return False
        elif response.status_code == 401:
            print("❌ Polygon API authentication failed - check your API key")
            return False
        elif response.status_code == 429:
            print("⚠️  Polygon API rate limit - but API key is valid")
            return True
        else:
            print(f"❌ Polygon API returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test Polygon API: {e}")
        return False

def main():
    """Main verification function"""
    print("🔍 BACKFILL SETUP VERIFICATION")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run all checks
    checks = [
        ("API Key", check_api_key()),
        ("InfluxDB", check_influxdb()),
        ("Directories", check_directories()),
        ("Dependencies", check_dependencies())
    ]
    
    # Test Polygon API if basic checks pass
    basic_checks_passed = all(result for _, result in checks)
    if basic_checks_passed:
        api_test = asyncio.run(test_polygon_api())
        checks.append(("Polygon API Test", api_test))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SETUP VERIFICATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:<20} {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("\n🚀 Ready to start aggressive backfill:")
        print("   python start_aggressive_backfill.py")
        print("\n📊 Expected results:")
        print("   • 3 years of 1-minute SPY data")
        print("   • 3 years of 1-minute QQQ data") 
        print("   • ~3 million data points total")
        print("   • 2-3 hours execution time")
        print("   • $0 cost (free tier)")
    else:
        print("❌ SETUP ISSUES FOUND")
        print("\n🔧 Please fix the failed checks above before running backfill")
        print("💡 See BACKFILL_SETUP_GUIDE.md for detailed instructions")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)