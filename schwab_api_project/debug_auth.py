"""
Schwab API Debug Script
Tests authentication and provides detailed error information
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import schwab
from schwab.auth import client_from_login_flow
import webbrowser

# Load environment variables
load_dotenv()

def test_authentication():
    """Test Schwab API authentication with detailed debugging"""
    
    print("=== Schwab API Authentication Test ===\n")
    
    # 1. Check environment variables
    api_key = os.getenv('SCHWAB_APP_KEY')
    app_secret = os.getenv('SCHWAB_APP_SECRET')
    callback_url = os.getenv('SCHWAB_CALLBACK_URL', 'https://127.0.0.1:8182')
    
    print("1. Environment Variables:")
    print(f"   API Key: {'*' * 20}...{api_key[-4:] if api_key and len(api_key) > 4 else 'NOT SET'}")
    print(f"   App Secret: {'*' * 10}...{app_secret[-4:] if app_secret and len(app_secret) > 4 else 'NOT SET'}")
    print(f"   Callback URL: {callback_url}")
    
    if not api_key or not app_secret:
        print("\n[ERROR] Missing credentials in .env file!")
        return False
    
    print("\n2. Testing Authentication...")
    print("   NOTE: A browser will open. You MUST:")
    print("   - Accept the security warning about the certificate")
    print("   - Log in with your Schwab credentials")
    print("   - Complete the authentication process\n")
    
    try:
        # Attempt authentication
        client = client_from_login_flow(
            api_key,
            app_secret,
            callback_url,
            webbrowser,
            callback_timeout=180,  # 3 minutes timeout
            interactive=True  # Show prompts
        )
        
        print("\n[SUCCESS] Authentication successful!")
        
        # Test with a simple API call
        print("\n3. Testing API Access...")
        response = client.get_quote('SPY')
        
        if response.status_code == 200:
            print("[SUCCESS] API call successful! Can retrieve market data.")
            return True
        else:
            print(f"[ERROR] API call failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        error_msg = str(e)
        print(f"\n[ERROR] Authentication failed: {error_msg}")
        
        # Provide specific troubleshooting based on error
        if "invalid_client" in error_msg:
            print("\n=== TROUBLESHOOTING 'invalid_client' ERROR ===")
            print("This error usually means one of the following:")
            print("\n1. APP NOT READY (MOST COMMON):")
            print("   - Log into https://developer.schwab.com/")
            print("   - Check your app status")
            print("   - If it says 'Approved - Pending', you MUST wait 1-3 days")
            print("   - Only 'Ready for Use' apps can authenticate")
            
            print("\n2. CALLBACK URL MISMATCH:")
            print("   - Your app's callback URL in Schwab MUST be: " + callback_url)
            print("   - No trailing slash allowed!")
            
            print("\n3. WRONG API PRODUCT:")
            print("   - Your app must have 'Accounts and Trading Production' selected")
            
            print("\n4. INCORRECT CREDENTIALS:")
            print("   - Double-check your API key and secret")
            print("   - Make sure there are no extra spaces or quotes")
            
        elif "timeout" in error_msg.lower():
            print("\n=== TROUBLESHOOTING TIMEOUT ERROR ===")
            print("1. Did you accept the security warning in the browser?")
            print("2. Did you complete the login process?")
            print("3. Try increasing the timeout or running again")
            
        return False

if __name__ == "__main__":
    print("Starting Schwab authentication test...\n")
    
    if test_authentication():
        print("\n=== ALL TESTS PASSED ===")
        print("Your Schwab API setup is working correctly!")
        print("You can now run your main.py script.")
    else:
        print("\n=== TESTS FAILED ===")
        print("Please fix the issues above and try again.")
