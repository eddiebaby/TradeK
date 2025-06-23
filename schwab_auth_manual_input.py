#!/usr/bin/env python3
"""
Manual Schwab Authentication - Interactive Version
"""

import os
import requests
import base64
import hashlib
import webbrowser
import time
from dotenv import load_dotenv, set_key, find_dotenv
from urllib.parse import urlparse, parse_qs

def generate_pkce_challenge():
    """Generates a PKCE code verifier and challenge."""
    code_verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b'=').decode('utf-8')
    code_challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode('utf-8')).digest()).rstrip(b'=').decode('utf-8')
    return code_verifier, code_challenge

def get_tokens(auth_code, code_verifier, app_key, secret, callback_url):
    """Exchanges the authorization code for access and refresh tokens."""
    token_url = "https://api.schwabapi.com/v1/oauth/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {base64.b64encode(f'{app_key}:{secret}'.encode()).decode()}"
    }
    data = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": callback_url,
        "code_verifier": code_verifier
    }
    response = requests.post(token_url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()

def save_tokens_to_env(tokens):
    """Saves access and refresh tokens to the .env file."""
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    expires_in = tokens.get("expires_in", 1800)  # Default 30 minutes

    if not access_token or not refresh_token:
        print("Error: Failed to retrieve tokens.")
        print(f"Response: {tokens}")
        return

    dotenv_file = find_dotenv() or '.env'
    set_key(dotenv_file, "SCHWAB_ACCESS_TOKEN", access_token)
    set_key(dotenv_file, "SCHWAB_REFRESH_TOKEN", refresh_token)
    
    # Save token expiration time
    expiry_time = int(time.time()) + expires_in - 60  # 1 minute buffer
    set_key(dotenv_file, "SCHWAB_TOKEN_EXPIRY", str(expiry_time))
    
    print("\nSuccessfully retrieved and saved tokens to .env file!")
    print(f"Access Token (expires in {expires_in}s): {access_token[:15]}...")
    print(f"Refresh Token: {refresh_token[:15]}...")

def main():
    """Main function to handle manual authentication flow."""
    load_dotenv()
    
    app_key = os.getenv("SCHWAB_APP_KEY")
    secret = os.getenv("SCHWAB_APP_SECRET")
    callback_url = os.getenv("SCHWAB_CALLBACK_URL")

    if not all([app_key, secret, callback_url]):
        print("Error: Required credentials are missing.")
        print("Ensure SCHWAB_APP_KEY, SCHWAB_APP_SECRET, and SCHWAB_CALLBACK_URL are set in .env")
        return

    print("🔐 Schwab Manual Authentication")
    print("=" * 40)
    print(f"App Key: {app_key[:8]}...")
    print(f"Callback URL: {callback_url}")
    
    # Generate PKCE challenge
    code_verifier, code_challenge = generate_pkce_challenge()
    
    # Build authorization URL
    auth_url = (
        f"https://api.schwabapi.com/v1/oauth/authorize?"
        f"client_id={app_key}&"
        f"redirect_uri={callback_url}&"
        f"response_type=code&"
        f"code_challenge={code_challenge}&"
        f"code_challenge_method=S256"
    )
    
    print("\n🔗 Authorization URL:")
    print(auth_url)
    print("\n📱 Steps:")
    print("1. Click the URL above")
    print("2. Log in to Schwab")
    print("3. Authorize the application")
    print("4. Copy the callback URL you get redirected to")
    print("5. Paste it below")
    
    # Manual input for callback URL
    print("\n" + "=" * 60)
    callback_response = input("Paste the callback URL here: ").strip()
    
    if not callback_response.startswith(callback_url):
        print(f"Error: Invalid callback URL. Must start with {callback_url}")
        return
    
    # Parse callback URL
    parsed_url = urlparse(callback_response)
    query_params = parse_qs(parsed_url.query)
    auth_code = query_params.get("code", [None])[0]

    if not auth_code:
        print("Error: Could not find 'code' in the redirected URL.")
        return

    print(f"\n✅ Authorization code extracted: {auth_code[:20]}...")
    
    try:
        # Exchange code for tokens
        print("🔄 Exchanging authorization code for tokens...")
        tokens = get_tokens(auth_code, code_verifier, app_key, secret, callback_url)
        save_tokens_to_env(tokens)
        
        print("\n🎉 Schwab authentication completed successfully!")
        print("✅ Tokens saved to .env file")
        print("🔐 Ready for Schwab API integration")
        
    except Exception as e:
        print(f"\n❌ Token exchange failed: {e}")

if __name__ == "__main__":
    main()