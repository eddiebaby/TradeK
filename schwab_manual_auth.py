#!/usr/bin/env python3
"""
Manual Schwab API Authentication
Provides authorization URL for manual browser completion.
"""

import os
import json
from dotenv import load_dotenv
import schwab.auth
from urllib.parse import urlparse, parse_qs

# Load environment variables
load_dotenv()

def main():
    """Manual authentication flow"""
    print("🚀 Schwab API Manual Authentication")
    print("=" * 50)
    
    # Get environment variables
    app_key = os.getenv('SCHWAB_APP_KEY')
    app_secret = os.getenv('SCHWAB_SECRET')
    redirect_uri = os.getenv('SCHWAB_REDIRECT_URI', 'https://127.0.0.1:8182')
    
    if not app_key or not app_secret:
        print("❌ Missing SCHWAB_APP_KEY or SCHWAB_SECRET environment variables")
        return
    
    print(f"📋 App Key: {app_key[:8]}...")
    print(f"📋 Redirect URI: {redirect_uri}")
    
    # Generate the authorization URL manually
    from urllib.parse import urlencode
    import secrets
    
    # Generate state for security
    state = secrets.token_urlsafe(32)
    
    auth_params = {
        'response_type': 'code',
        'client_id': app_key,
        'redirect_uri': redirect_uri,
        'state': state
    }
    
    auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?{urlencode(auth_params)}"
    
    print(f"\n" + "=" * 70)
    print("📱 MANUAL AUTHENTICATION STEPS")
    print("=" * 70)
    print(f"\n1. Open this URL in your browser:")
    print(f"   {auth_url}")
    print(f"\n2. Log in to your Schwab account")
    print(f"3. Select/authorize the accounts you want to access")
    print(f"4. After authorization, you'll be redirected to a URL that starts with:")
    print(f"   {redirect_uri}")
    print(f"5. The page won't load (this is normal) - just copy the ENTIRE URL")
    print(f"6. Paste the full callback URL below")
    print(f"\n" + "=" * 70)
    
    # Get the callback URL from user
    callback_url = input("\nPaste the full callback URL here: ").strip()
    
    if not callback_url.startswith(redirect_uri):
        print(f"❌ Invalid callback URL. Must start with {redirect_uri}")
        return
    
    # Parse the callback URL
    try:
        parsed_url = urlparse(callback_url)
        query_params = parse_qs(parsed_url.query)
        
        if 'code' not in query_params:
            print("❌ No authorization code found in callback URL")
            return
        
        if 'state' not in query_params:
            print("❌ No state parameter found in callback URL")
            return
        
        auth_code = query_params['code'][0]
        returned_state = query_params['state'][0]
        
        # Verify state
        if returned_state != state:
            print("❌ State mismatch - possible security issue")
            return
        
        print(f"✅ Authorization code extracted: {auth_code[:20]}...")
        
        # Now use the schwab library to exchange the code for tokens
        print(f"\n🔑 Exchanging authorization code for tokens...")
        
        try:
            client = schwab.auth.client_from_received_url(
                callback_url,
                app_key,
                app_secret,
                "schwab_tokens.json"
            )
            
            print(f"✅ Authentication successful!")
            print(f"✅ schwab_tokens.json created")
            
            # Test the connection
            try:
                response = client.get_account_numbers()
                accounts = response.json()
                print(f"✅ API test successful!")
                print(f"📊 Connected to {len(accounts)} account(s):")
                for account in accounts:
                    account_number = account.get('accountNumber', 'Unknown')
                    print(f"   • Account: {account_number}")
                
            except Exception as e:
                print(f"⚠️  API test warning: {e}")
            
            print(f"\n🎉 Schwab API is now ready for TradeKnowledge!")
            print(f"🔐 Tokens saved in schwab_tokens.json")
            print(f"🔄 Tokens will auto-refresh as needed")
            
        except Exception as e:
            print(f"❌ Token exchange failed: {e}")
            
    except Exception as e:
        print(f"❌ Error parsing callback URL: {e}")

if __name__ == "__main__":
    main()