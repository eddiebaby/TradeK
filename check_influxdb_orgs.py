#!/usr/bin/env python3
"""
Quick script to check available InfluxDB organizations and buckets
"""
import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

load_dotenv()

url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
token = os.getenv("INFLUXDB_TOKEN")

if not token:
    print("❌ INFLUXDB_TOKEN not found in environment")
    exit(1)

try:
    client = InfluxDBClient(url=url, token=token)
    
    # Get organizations
    orgs_api = client.organizations_api()
    orgs = orgs_api.find_organizations()
    
    print("🏢 Available Organizations:")
    for org in orgs:
        print(f"  - Name: {org.name}")
        print(f"    ID: {org.id}")
        
        # Get buckets for this org
        buckets_api = client.buckets_api()
        buckets = buckets_api.find_buckets(org_id=org.id)
        
        print(f"    Buckets:")
        for bucket in buckets.buckets:
            print(f"      - {bucket.name}")
        print()
    
    client.close()
    
except Exception as e:
    print(f"❌ Error: {e}")