#!/usr/bin/env python3
"""
Manual Schwab token creation - bypass the problematic authorization flow
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

def manual_token_exchange():
    """
    If you can get an authorization code manually, we can exchange it for tokens
    """
    client_id = os.getenv('SCHWAB_APP_KEY')
    client_secret = os.getenv('SCHWAB_SECRET')
    redirect_uri = os.getenv('SCHWAB_REDIRECT_URI')
    
    print("🔐 Manual Schwab Token Exchange")
    print("=" * 40)
    print(f"Client ID: {client_id}")
    print(f"Client Secret: {client_secret[:8]}...")
    print(f"Redirect URI: {redirect_uri}")
    
    print(f"\n🤔 The issue seems to be with the authorization URL generation.")
    print(f"Your authorization request is getting redirected to a different client ID.")
    print(f"Expected: {client_id}")
    print(f"Getting:  baf15750-fe85-4b9d-b276-99ab43502838")
    
    print(f"\n💡 Possible solutions:")
    print(f"1. Check if your Schwab app is fully approved and active")
    print(f"2. Verify the app key and secret are correct")
    print(f"3. Try logging into Schwab Developer Portal and regenerating credentials")
    print(f"4. Check if there are any pending approvals or setup steps")
    
    print(f"\n🔍 App Status Check:")
    print(f"Your app shows 'Ready For Use' which should be correct.")
    print(f"But the clientID mismatch suggests an issue.")
    
    # Try a direct token endpoint test (this will fail but show us error details)
    print(f"\n🧪 Testing token endpoint accessibility...")
    
    test_data = {
        'grant_type': 'authorization_code',
        'code': 'test_code',
        'client_id': client_id,
        'client_secret': client_secret,
        'redirect_uri': redirect_uri
    }
    
    try:
        response = requests.post(
            'https://api.schwabapi.com/v1/oauth/token',
            data=test_data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        print(f"Response status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Connection error: {e}")
    
    print(f"\n🎯 Recommendation:")
    print(f"1. Log into your Schwab Developer Portal")
    print(f"2. Go to your app: prod-scottschweizergmailcom-fc02ec60-9c51-4afa-a602-dcf262b75269")
    print(f"3. Check if there are any pending actions or approvals needed")
    print(f"4. Verify the App Key matches: {client_id}")
    print(f"5. Consider regenerating the credentials if needed")

if __name__ == "__main__":
    manual_token_exchange()