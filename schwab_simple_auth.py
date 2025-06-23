#!/usr/bin/env python3
"""
Schwab API Authentication Setup Script (Simple Version)
Uses schwab-py library for streamlined authentication flow.
"""

import os
import json
from dotenv import load_dotenv
import schwab.auth

# Load environment variables
load_dotenv()

def check_environment_variables():
    """Check if required environment variables are set"""
    print("🔍 Checking environment variables...")
    
    app_key = os.getenv('SCHWAB_APP_KEY')
    app_secret = os.getenv('SCHWAB_SECRET')
    
    issues = []
    if not app_key:
        issues.append("SCHWAB_APP_KEY environment variable is not set")
    else:
        print(f"✓ SCHWAB_APP_KEY is set: {app_key[:8]}...")
    
    if not app_secret:
        issues.append("SCHWAB_SECRET environment variable is not set")
    else:
        print(f"✓ SCHWAB_SECRET is set: {app_secret[:8]}...")
    
    return issues

def verify_token_file():
    """Verify that the token file was created and contains valid data"""
    token_file = "schwab_tokens.json"
    
    if not os.path.exists(token_file):
        print(f"❌ Token file {token_file} was not created")
        return False
    
    try:
        with open(token_file, 'r') as f:
            tokens = json.load(f)
        
        required_keys = ['access_token', 'refresh_token', 'token_type']
        missing_keys = [key for key in required_keys if key not in tokens]
        
        if missing_keys:
            print(f"❌ Token file is missing required keys: {missing_keys}")
            return False
        
        print(f"✅ Token file verified - contains all required authentication data")
        return True
        
    except Exception as e:
        print(f"❌ Error reading token file: {e}")
        return False

def main():
    """Main authentication setup process"""
    print("🚀 Schwab API Authentication Setup (Simple)")
    print("=" * 50)
    
    # Step 1: Check environment variables
    env_issues = check_environment_variables()
    if env_issues:
        print("\n❌ Environment Variable Issues:")
        for issue in env_issues:
            print(f"   • {issue}")
        print("\nPlease set the required environment variables before running this script.")
        return
    
    # Step 2: Get environment variables
    app_key = os.getenv('SCHWAB_APP_KEY')
    app_secret = os.getenv('SCHWAB_SECRET')
    redirect_uri = os.getenv('SCHWAB_REDIRECT_URI', 'https://127.0.0.1:8182')
    
    print(f"\n📋 Using configuration:")
    print(f"   App Key: {app_key[:8]}...")
    print(f"   Redirect URI: {redirect_uri}")
    
    print(f"\n📚 Authentication Process:")
    print(f"1. A browser window will open with Schwab login")
    print(f"2. Log in with your normal Schwab credentials")
    print(f"3. Authorize the TradeKnowledge application")
    print(f"4. You'll be redirected to a localhost URL that won't load (this is normal)")
    print(f"5. Copy the ENTIRE URL from your browser's address bar")
    print(f"6. Paste it when prompted")
    
    input("\nPress Enter when you're ready to continue...")
    
    try:
        print(f"\n🔑 Starting Schwab authentication flow...")
        
        # Use schwab-py library's built-in authentication flow
        client = schwab.auth.client_from_login_flow(
            app_key, 
            app_secret, 
            redirect_uri, 
            "schwab_tokens.json"  # This is where tokens will be saved
        )
        
        print("✅ Successfully created schwab_tokens.json")
        print("✅ Authentication setup complete!")
        
        # Verify token file
        if verify_token_file():
            print(f"\n🎉 Authentication setup completed successfully!")
            print(f"✅ schwab_tokens.json has been created")
            print(f"✅ You can now use Schwab API through TradeKnowledge")
            
            # Test the client
            try:
                print(f"\n🧪 Testing API connection...")
                # Get account info to verify the connection works
                response = client.get_account_numbers()
                print(f"✅ API connection successful!")
                print(f"📊 Found {len(response.json())} account(s)")
            except Exception as test_error:
                print(f"⚠️  API test failed: {test_error}")
                print(f"   (This might be normal if account permissions are limited)")
            
        else:
            print(f"\n❌ Token file verification failed")
            
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Make sure your Schwab app is configured correctly")
        print(f"2. Check that the redirect URI matches your app settings: {redirect_uri}")
        print(f"3. Ensure your Schwab account has API access enabled")

if __name__ == "__main__":
    main()