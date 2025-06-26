import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Schwab API Key
api_key = os.getenv('SCHWAB_APP_KEY')

# Get Schwab API Secret
app_secret = os.getenv('SCHWAB_APP_SECRET')

# Get Schwab Callback URL from environment
callback_url = os.getenv('SCHWAB_CALLBACK_URL')

def check_credentials(key, secret, url):
    """Checks if all required Schwab API credentials are present."""
    if not key or not secret or not url:
        raise ValueError("Missing one or more Schwab API credentials in .env file. "
                         "Please ensure SCHWAB_APP_KEY, SCHWAB_APP_SECRET, "
                         "and SCHWAB_CALLBACK_URL are set.")
    return True