#!/usr/bin/env python3
"""
Debug Schwab Authentication Issues
Try multiple redirect URI formats to find what works.
"""

import os
from dotenv import load_dotenv
from urllib.parse import urlencode
import secrets

load_dotenv()

def main():
    app_key = os.getenv('SCHWAB_APP_KEY')
    app_secret = os.getenv('SCHWAB_SECRET')
    
    print("🔍 Schwab Authentication Debug")
    print("=" * 40)
    print(f"App Key: {app_key[:8]}...")
    print(f"App Secret: {app_secret[:8]}...")
    
    # Common redirect URI formats for Schwab
    redirect_uris = [
        "https://localhost:8080",
        "https://127.0.0.1:8080", 
        "https://localhost:8182",
        "https://127.0.0.1:8182",
        "https://localhost:8080/",
        "https://127.0.0.1:8080/",
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ]
    
    print(f"\n🧪 Testing different redirect URI formats:")
    print(f"Note: You configured this in your Schwab Developer Portal app")
    print(f"Try each URL below to see which one works:\n")
    
    for i, redirect_uri in enumerate(redirect_uris, 1):
        state = secrets.token_urlsafe(16)
        
        auth_params = {
            'response_type': 'code',
            'client_id': app_key,
            'redirect_uri': redirect_uri,
            'state': state,
            'scope': 'readonly'
        }
        
        auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?{urlencode(auth_params)}"
        
        print(f"Option {i}: {redirect_uri}")
        print(f"URL: {auth_url}")
        print(f"State: {state}")
        print(f"-" * 60)
    
    print(f"\n💡 Troubleshooting Tips:")
    print(f"1. Check your Schwab Developer Portal app configuration")
    print(f"2. The redirect URI must EXACTLY match what you configured")
    print(f"3. Common working formats:")
    print(f"   • https://localhost:8080 (most common)")
    print(f"   • http://localhost:8080 (if HTTPS not required)")
    print(f"4. If none work, you may need to update your app configuration")
    
    print(f"\n🔧 Next Steps:")
    print(f"1. Try each URL above in your browser")
    print(f"2. Look for the one that doesn't give a 400 error")
    print(f"3. Update your .env file with the working redirect URI")
    print(f"4. Let me know which one works!")

if __name__ == "__main__":
    main()