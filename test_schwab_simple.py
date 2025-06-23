#!/usr/bin/env python3
"""
Test simple schwab authentication to see what works
"""

import os
from dotenv import load_dotenv

load_dotenv()

app_key = os.getenv('SCHWAB_APP_KEY')
app_secret = os.getenv('SCHWAB_SECRET')

print(f"App Key: {app_key}")
print(f"App Secret: {app_secret[:8]}...")

# Try with localhost instead of 127.0.0.1
redirect_uri = "https://localhost:8182"

print(f"Trying redirect URI: {redirect_uri}")

import schwab.auth

try:
    print("Starting authentication with localhost...")
    client = schwab.auth.client_from_login_flow(
        app_key,
        app_secret, 
        redirect_uri,
        "schwab_tokens.json"
    )
    print("SUCCESS!")
except Exception as e:
    print(f"Failed: {e}")