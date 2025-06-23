import os
import requests
import base64
import hashlib
import webbrowser
# import pyperclip  # Commented out to avoid dependency issues
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

def refresh_tokens(app_key, secret, refresh_token):
    """Refreshes the access token using the refresh token."""
    token_url = "https://api.schwabapi.com/v1/oauth/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {base64.b64encode(f'{app_key}:{secret}'.encode()).decode()}"
    }
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
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

def is_token_valid():
    """Checks if the current access token is still valid."""
    load_dotenv()
    expiry_time = os.getenv("SCHWAB_TOKEN_EXPIRY")
    
    if not expiry_time:
        return False
    
    try:
        return int(time.time()) < int(expiry_time)
    except ValueError:
        return False

def get_valid_access_token():
    """Returns a valid access token, refreshing if necessary."""
    load_dotenv()
    
    if is_token_valid():
        return os.getenv("SCHWAB_ACCESS_TOKEN")
    
    # Token expired, refresh it
    app_key = os.environ.get("SCHWAB_APP_KEY") or os.getenv("SCHWAB_APP_KEY")
    secret = os.environ.get("SCHWAB_APP_SECRET") or os.getenv("SCHWAB_SECRET")
    refresh_token_from_env = os.environ.get("SCHWAB_REFRESH_TOKEN") or os.getenv("SCHWAB_REFRESH_TOKEN")
    
    if not all([app_key, secret, refresh_token_from_env]):
        print("Error: Cannot refresh token - missing credentials")
        return None
    
    try:
        print("Access token expired, refreshing...")
        new_tokens = refresh_tokens(app_key, secret, refresh_token_from_env)
        save_tokens_to_env(new_tokens)
        return new_tokens.get("access_token")
    except Exception as e:
        print(f"Error refreshing token: {e}")
        return None

def make_authenticated_request(url, method="GET", **kwargs):
    """Makes an authenticated request, automatically refreshing tokens if needed."""
    access_token = get_valid_access_token()
    
    if not access_token:
        raise Exception("No valid access token available")
    
    headers = kwargs.get('headers', {})
    headers['Authorization'] = f'Bearer {access_token}'
    kwargs['headers'] = headers
    
    if method.upper() == "GET":
        return requests.get(url, **kwargs)
    elif method.upper() == "POST":
        return requests.post(url, **kwargs)
    elif method.upper() == "PUT":
        return requests.put(url, **kwargs)
    elif method.upper() == "DELETE":
        return requests.delete(url, **kwargs)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

def main():
    """Main function to handle the manual authentication flow."""
    load_dotenv()  # Load from .env file as fallback
    
    # Prioritize system environment variables over .env file
    app_key = os.environ.get("SCHWAB_APP_KEY") or os.getenv("SCHWAB_APP_KEY")
    secret = os.environ.get("SCHWAB_APP_SECRET") or os.getenv("SCHWAB_SECRET")
    callback_url = os.environ.get("SCHWAB_CALLBACK_URL") or os.getenv("SCHWAB_CALLBACK_URL")
    refresh_token_from_env = os.environ.get("SCHWAB_REFRESH_TOKEN") or os.getenv("SCHWAB_REFRESH_TOKEN")

    if not all([app_key, secret, callback_url]):
        print("Error: Required credentials are missing.")
        print("Ensure SCHWAB_APP_KEY, SCHWAB_APP_SECRET, and SCHWAB_CALLBACK_URL are set as:")
        print("1. System environment variables (recommended for security)")
        print("2. Or in your .env file (for development only)")
        return

    print("Choose an action:")
    print("1. Full Authentication")
    print("2. Refresh Tokens")
    choice = "1"  # Auto-select full authentication
    print(f"Auto-selecting option: {choice}")

    try:
        if choice == '1':
            # Full authentication flow
            code_verifier, code_challenge = generate_pkce_challenge()
            auth_url = (
                f"https://api.schwabapi.com/v1/oauth/authorize?client_id={app_key}"
                f"&redirect_uri={callback_url}&response_type=code&code_challenge={code_challenge}"
                f"&code_challenge_method=S256"
            )
            print("\nGenerated Authentication URL:")
            print(auth_url)
            print("\nPlease copy the URL manually.")
            webbrowser.open(auth_url)
            print("\nAfter authorizing in your browser, you'll get a callback URL.")
            print("Paste that callback URL when ready...")
            redirected_url = input("\nPaste the redirected URL here and press Enter:\n")
            parsed_url = urlparse(redirected_url)
            query_params = parse_qs(parsed_url.query)
            auth_code = query_params.get("code", [None])[0]

            if not auth_code:
                print("Error: Could not find 'code' in the redirected URL.")
                return

            print(f"\nExtracted Authorization Code: {auth_code}")
            tokens = get_tokens(auth_code, code_verifier, app_key, secret, callback_url)
            save_tokens_to_env(tokens)

        elif choice == '2':
            # Refresh tokens flow
            if not refresh_token_from_env:
                print("Error: SCHWAB_REFRESH_TOKEN not found in .env file. Please run full authentication first.")
                return
            print("\nRefreshing tokens...")
            new_tokens = refresh_tokens(app_key, secret, refresh_token_from_env)
            save_tokens_to_env(new_tokens)
        
        else:
            print("Invalid choice. Please enter 1 or 2.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
