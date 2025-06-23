#!/usr/bin/env python3
"""
Schwab OAuth Authentication - Official Implementation
Following Schwab's developer documentation exactly.
"""

import os
import json
import base64
import hashlib
import secrets
import webbrowser
from urllib.parse import urlencode, urlparse, parse_qs
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

class SchwabOAuth:
    """
    Official Schwab OAuth implementation following their developer guide.
    """
    
    def __init__(self):
        self.client_id = os.getenv('SCHWAB_APP_KEY')
        self.client_secret = os.getenv('SCHWAB_SECRET')
        self.redirect_uri = os.getenv('SCHWAB_REDIRECT_URI', 'https://localhost:8080')
        
        # Schwab OAuth endpoints from their documentation
        self.auth_url = "https://api.schwabapi.com/v1/oauth/authorize"
        self.token_url = "https://api.schwabapi.com/v1/oauth/token"
        
        if not self.client_id or not self.client_secret:
            raise ValueError("SCHWAB_APP_KEY and SCHWAB_SECRET must be set in environment")
    
    def generate_pkce_challenge(self):
        """Generate PKCE challenge for enhanced security"""
        # Generate code verifier (43-128 characters)
        code_verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(96)
        ).decode('utf-8').rstrip('=')
        
        # Generate code challenge
        challenge = hashlib.sha256(code_verifier.encode('utf-8')).digest()
        code_challenge = base64.urlsafe_b64encode(challenge).decode('utf-8').rstrip('=')
        
        return code_verifier, code_challenge
    
    def get_authorization_url(self):
        """
        Step 1: Generate authorization URL following Schwab's specifications
        """
        # Generate PKCE challenge
        code_verifier, code_challenge = self.generate_pkce_challenge()
        
        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)
        
        # Store these for later use
        self.code_verifier = code_verifier
        self.state = state
        
        # Build authorization URL per Schwab's documentation
        auth_params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': 'readonly',  # Standard scope for account access
            'state': state,
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256'
        }
        
        auth_url = f"{self.auth_url}?{urlencode(auth_params)}"
        
        print("🔐 Schwab OAuth Authentication")
        print("=" * 50)
        print(f"Client ID: {self.client_id[:8]}...")
        print(f"Redirect URI: {self.redirect_uri}")
        print(f"State: {state}")
        print("=" * 50)
        
        return auth_url
    
    def exchange_code_for_tokens(self, callback_url):
        """
        Step 2: Exchange authorization code for access tokens
        """
        # Parse callback URL
        parsed_url = urlparse(callback_url)
        query_params = parse_qs(parsed_url.query)
        
        # Extract authorization code and state
        if 'code' not in query_params:
            raise ValueError("No authorization code found in callback URL")
        
        if 'state' not in query_params:
            raise ValueError("No state parameter found in callback URL")
        
        auth_code = query_params['code'][0]
        returned_state = query_params['state'][0]
        
        # Verify state parameter (CSRF protection)
        if returned_state != self.state:
            raise ValueError("State parameter mismatch - possible CSRF attack")
        
        # Prepare token request per Schwab's specification
        token_data = {
            'grant_type': 'authorization_code',
            'code': auth_code,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'redirect_uri': self.redirect_uri,
            'code_verifier': self.code_verifier
        }
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        
        print(f"🔄 Exchanging authorization code for tokens...")
        print(f"Authorization code: {auth_code[:20]}...")
        
        # Make token exchange request
        response = requests.post(
            self.token_url,
            data=token_data,
            headers=headers
        )
        
        if response.status_code != 200:
            error_msg = f"Token exchange failed: {response.status_code} - {response.text}"
            raise ValueError(error_msg)
        
        tokens = response.json()
        
        # Save tokens to file
        token_file = "schwab_tokens.json"
        with open(token_file, 'w') as f:
            json.dump(tokens, f, indent=2)
        
        print(f"✅ Tokens successfully obtained and saved to {token_file}")
        print(f"Access Token: {tokens['access_token'][:20]}...")
        print(f"Refresh Token: {tokens['refresh_token'][:20]}...")
        print(f"Expires in: {tokens['expires_in']} seconds")
        
        return tokens
    
    def test_api_access(self, access_token):
        """
        Step 3: Test API access with the new token
        """
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }
        
        # Test with account numbers endpoint
        test_url = "https://api.schwabapi.com/trader/v1/accounts/accountNumbers"
        
        print(f"\n🧪 Testing API access...")
        
        response = requests.get(test_url, headers=headers)
        
        if response.status_code == 200:
            accounts = response.json()
            print(f"✅ API test successful!")
            print(f"📊 Found {len(accounts)} account(s):")
            for account in accounts:
                account_number = account.get('accountNumber', 'Unknown')
                print(f"   • Account: {account_number}")
            return True
        else:
            print(f"⚠️ API test failed: {response.status_code} - {response.text}")
            return False

def main():
    """
    Main authentication flow following Schwab's official documentation
    """
    try:
        # Initialize OAuth client
        oauth = SchwabOAuth()
        
        # Step 1: Get authorization URL
        auth_url = oauth.get_authorization_url()
        
        print(f"\n📱 STEP 1: User Authorization")
        print(f"-" * 30)
        print(f"1. Open this URL in your browser:")
        print(f"   {auth_url}")
        print(f"\n2. Log in to your Schwab account")
        print(f"3. Grant consent and select accounts to authorize")
        print(f"4. You'll be redirected to: {oauth.redirect_uri}")
        print(f"5. Copy the ENTIRE callback URL and paste it below")
        
        # Get callback URL from user
        print(f"\n" + "=" * 60)
        callback_url = input("Paste the full callback URL here: ").strip()
        
        if not callback_url.startswith(oauth.redirect_uri):
            print(f"❌ Invalid callback URL. Must start with {oauth.redirect_uri}")
            return
        
        # Step 2: Exchange code for tokens
        tokens = oauth.exchange_code_for_tokens(callback_url)
        
        # Step 3: Test API access
        oauth.test_api_access(tokens['access_token'])
        
        print(f"\n🎉 Schwab OAuth authentication completed successfully!")
        print(f"✅ Tokens saved to schwab_tokens.json")
        print(f"🔐 Ready for Schwab API integration")
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"1. Verify your app configuration in Schwab Developer Portal")
        print(f"2. Ensure redirect URI exactly matches your app settings")
        print(f"3. Check that your app is approved and active")

if __name__ == "__main__":
    main()