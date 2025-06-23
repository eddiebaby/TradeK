#!/usr/bin/env python3
"""
Generate simple Schwab authorization URL without any special encoding
"""

import os
from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv('SCHWAB_APP_KEY')
redirect_uri = "https://127.0.0.1:8182"

# Simple URL construction
auth_url = (
    f"https://api.schwabapi.com/v1/oauth/authorize"
    f"?response_type=code"
    f"&client_id={client_id}"
    f"&redirect_uri={redirect_uri}"
    f"&scope=readonly"
)

print("🔗 Simple Schwab Authorization URL:")
print(auth_url)
print(f"\nRedirect URI: {redirect_uri}")
print(f"Client ID: {client_id}")