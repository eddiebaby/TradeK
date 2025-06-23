#!/usr/bin/env python3
"""
Direct Schwab API Authentication
Uses schwab-py library with immediate authentication flow.
"""

import os
import json
from dotenv import load_dotenv
import schwab.auth

# Load environment variables
load_dotenv()

def main():
    """Start immediate authentication"""
    print("🚀 Schwab API Direct Authentication")
    print("=" * 40)
    
    # Get environment variables
    app_key = os.getenv('SCHWAB_APP_KEY')
    app_secret = os.getenv('SCHWAB_SECRET')
    redirect_uri = os.getenv('SCHWAB_REDIRECT_URI', 'https://127.0.0.1:8182')
    
    if not app_key or not app_secret:
        print("❌ Missing SCHWAB_APP_KEY or SCHWAB_SECRET environment variables")
        return
    
    print(f"📋 App Key: {app_key[:8]}...")
    print(f"📋 Redirect URI: {redirect_uri}")
    print(f"\n🌐 Starting authentication flow...")
    print(f"   A browser window will open for Schwab login")
    print(f"   Follow the prompts in the browser and console")
    
    try:
        # Start authentication flow (non-interactive)
        client = schwab.auth.client_from_login_flow(
            app_key, 
            app_secret, 
            redirect_uri, 
            "schwab_tokens.json",
            interactive=False
        )
        
        print(f"\n✅ Authentication successful!")
        print(f"✅ schwab_tokens.json created")
        
        # Test the connection
        try:
            response = client.get_account_numbers()
            print(f"✅ API test successful - found {len(response.json())} account(s)")
        except Exception as e:
            print(f"⚠️  API test warning: {e}")
        
        print(f"\n🎉 Schwab API is now ready for TradeKnowledge!")
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")

if __name__ == "__main__":
    main()