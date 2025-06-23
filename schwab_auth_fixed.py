#!/usr/bin/env python3
"""
Fixed Schwab API Authentication
Uses proper redirect URI formatting for Schwab OAuth.
"""

import os
import json
from dotenv import load_dotenv
import schwab.auth
from urllib.parse import urlparse, parse_qs, quote

# Load environment variables
load_dotenv()

def main():
    """Fixed authentication flow"""
    print("🚀 Schwab API Authentication (Fixed)")
    print("=" * 45)
    
    # Get environment variables
    app_key = os.getenv('SCHWAB_APP_KEY')
    app_secret = os.getenv('SCHWAB_SECRET')
    redirect_uri = os.getenv('SCHWAB_REDIRECT_URI', 'https://127.0.0.1:8182')
    
    if not app_key or not app_secret:
        print("❌ Missing SCHWAB_APP_KEY or SCHWAB_SECRET environment variables")
        return
    
    print(f"📋 App Key: {app_key[:8]}...")
    print(f"📋 Redirect URI: {redirect_uri}")
    
    # Try different redirect URI formats that Schwab might accept
    redirect_options = [
        'https://127.0.0.1:8182',
        'https://localhost:8182',
        'https://127.0.0.1:8182/',
        'https://localhost:8182/'
    ]
    
    print(f"\n🔧 Testing different redirect URI formats...")
    
    for i, test_redirect in enumerate(redirect_options, 1):
        print(f"\n📌 Option {i}: {test_redirect}")
        
        # Generate the authorization URL manually with proper encoding
        from urllib.parse import urlencode
        import secrets
        
        # Generate state for security
        state = secrets.token_urlsafe(32)
        
        auth_params = {
            'response_type': 'code',
            'client_id': app_key,
            'redirect_uri': test_redirect,
            'state': state,
            'scope': 'readonly'  # Add explicit scope
        }
        
        auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?{urlencode(auth_params, safe=':/?#[]@!$&()*+,;=')}"
        
        print(f"🔗 Authorization URL:")
        print(f"   {auth_url}")
        
        print(f"\n📱 Try this URL in your browser:")
        print(f"1. Click or copy the URL above")
        print(f"2. Log in to Schwab and authorize the app")
        print(f"3. If you get redirected successfully (even if page doesn't load), copy the callback URL")
        print(f"4. If you get an error, try the next option")
        
        # Ask user if this worked
        response = input(f"\nDid this URL work? (y/n/callback_url): ").strip()
        
        if response.lower() == 'y':
            callback_url = input("Paste the callback URL here: ").strip()
            return process_callback(callback_url, app_key, app_secret, state)
        elif response.lower().startswith('https://'):
            # User pasted the callback URL directly
            return process_callback(response, app_key, app_secret, state)
        elif response.lower() == 'n':
            print("❌ This URL didn't work, trying next option...")
            continue
        else:
            print("❌ Invalid response, trying next option...")
            continue
    
    print(f"\n❌ None of the redirect URI formats worked.")
    print(f"💡 Suggestions:")
    print(f"1. Check your Schwab app configuration in the developer portal")
    print(f"2. Verify the redirect URI exactly matches what's configured")
    print(f"3. Make sure your app is approved and active")

def process_callback(callback_url, app_key, app_secret, expected_state):
    """Process the callback URL and create tokens"""
    try:
        # Parse the callback URL
        parsed_url = urlparse(callback_url)
        query_params = parse_qs(parsed_url.query)
        
        if 'code' not in query_params:
            print("❌ No authorization code found in callback URL")
            return False
        
        if 'state' not in query_params:
            print("❌ No state parameter found in callback URL") 
            return False
        
        auth_code = query_params['code'][0]
        returned_state = query_params['state'][0]
        
        # Verify state
        if returned_state != expected_state:
            print("❌ State mismatch - possible security issue")
            return False
        
        print(f"✅ Authorization code extracted: {auth_code[:20]}...")
        
        # Exchange code for tokens using schwab library
        print(f"\n🔑 Exchanging authorization code for tokens...")
        
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
        return True
        
    except Exception as e:
        print(f"❌ Error processing callback: {e}")
        return False

if __name__ == "__main__":
    main()