# Schwab API Connection: Specification

This document outlines the project structure, configuration management, and authentication logic for a Python application connecting to the Schwab API.

## 1. Project Structure

The project will be organized into a main application directory and a separate directory for tests. This promotes modularity and separation of concerns.

```
/
|-- .env
|-- requirements.txt
|-- schwab_api_project/
|   |-- __init__.py
|   |-- config.py
|   |-- main.py
|-- tests/
|   |-- __init__.py
|   |-- test_config.py
|   |-- test_main.py
```

- **`.env`**: Stores sensitive credentials. Not committed to version control.
- **`requirements.txt`**: Lists project dependencies.
- **`schwab_api_project/`**: The main application package.
- **`schwab_api_project/config.py`**: Manages API configuration.
- **`schwab_api_project/main.py`**: Contains the core authentication and application logic.
- **`tests/`**: Contains all tests.
- **`tests/test_config.py`**: Tests for the configuration module.
- **`tests/test_main.py`**: Tests for the main application logic.

## 2. Configuration Management (`config.py`)

This module is responsible for loading API credentials from the `.env` file. It ensures that no secrets are hard-coded.

**SAPPO :Pattern:** Externalized Configuration
**SAPPO :ComponentRole:** Configuration Manager

### Pseudocode: [`schwab_api_project/config.py`](schwab_api_project/config.py)

```python
# PSEUDOCODE for schwab_api_project/config.py

import os
from dotenv import load_dotenv

# TDD-ANCHOR: test_env_file_loading
# Load environment variables from .env file
# load_dotenv()

# TDD-ANCHOR: test_api_key_presence
# Get Schwab API Key
# api_key = os.getenv('SCHWAB_APP_KEY')

# TDD-ANCHOR: test_api_secret_presence
# Get Schwab API Secret
# app_secret = os.getenv('SCHWAB_APP_SECRET')

# TDD-ANCHOR: test_callback_url_presence
# Get Schwab Callback URL
# callback_url = os.getenv('SCHWAB_CALLBACK_URL')

# TDD-ANCHOR: test_missing_credentials_handling
# FUNCTION check_credentials(api_key, app_secret, callback_url):
#     IF not api_key or not app_secret or not callback_url:
#         RAISE ValueError("Missing one or more Schwab API credentials in .env file")
#     RETURN True

# check_credentials(api_key, app_secret, callback_url)
```

## 3. Authentication Logic (`main.py`)

This is the main entry point of the application. It handles the OAuth 2.0 flow and provides a simple function to test the connection.

**SAPPO :Pattern:** API Gateway
**SAPPO :ComponentRole:** Authentication Handler

### Pseudocode: [`schwab_api_project/main.py`](schwab_api_project/main.py)

```python
# PSEUDOCODE for schwab_api_project/main.py

import schwab_api_project.config as config
from schwab.auth import client_from_token_file, APIClient
from pathlib import Path
import json

# Define path for the token file
# token_path = Path('token.json')

# TDD-ANCHOR: test_authentication_flow
# FUNCTION perform_authentication(api_key, app_secret, callback_url, token_path):
#     # Try to load the client from the token file
#     # client = client_from_token_file(token_path, api_key, app_secret)
#
#     # IF client.session.authorized():
#     #     PRINT "Authentication successful from token."
#     #     RETURN client
#
#     # If token is invalid or not present, start the manual auth flow
#     # client = APIClient(api_key, app_secret, callback_url)
#     # PRINT "Please go to the following URL and grant access:"
#     # PRINT client.auth_url()
#     # redirect_url = INPUT("Paste the redirect URL here: ")
#
#     # Exchange the authorization code for a token
#     # client.session.fetch_token(authorization_response=redirect_url)
#
#     # Save the token
#     # with open(token_path, 'w') as f:
#     #     json.dump(client.session.token, f)
#     # PRINT "Token saved successfully."
#
#     # RETURN client

# TDD-ANCHOR: test_get_account_numbers
# FUNCTION get_account_numbers(client):
#     # response = client.get_account_numbers()
#     # IF response.ok:
#     #     RETURN response.json()
#     # ELSE:
#     #     RAISE Exception(f"Failed to fetch account numbers: {response.text}")

# TDD-ANCHOR: test_main_execution_logic
# FUNCTION main():
#     # client = perform_authentication(config.api_key, config.app_secret, config.callback_url, token_path)
#     # account_numbers = get_account_numbers(client)
#     # PRINT "Successfully fetched account numbers:"
#     # PRINT account_numbers

# IF __name__ == "__main__":
#     main()
```

## 4. Dependencies (`requirements.txt`)

The following libraries are required for this project.

### File Content: [`requirements.txt`](requirements.txt)
```
python-dotenv
schwab-py
```

## 5. TDD Anchors Summary

- `test_env_file_loading`: Verify that the `.env` file is correctly loaded.
- `test_api_key_presence`: Ensure `SCHWAB_APP_KEY` is loaded.
- `test_api_secret_presence`: Ensure `SCHWAB_APP_SECRET` is loaded.
- `test_callback_url_presence`: Ensure `SCHWAB_CALLBACK_URL` is loaded.
- `test_missing_credentials_handling`: Test that the application raises an error if credentials are missing.
- `test_authentication_flow`: Mock the `schwab-py` library to test the full authentication and token-saving process.
- `test_get_account_numbers`: Test the function that fetches account numbers with a mock client.
- `test_main_execution_logic`: Test the main function's orchestration of the authentication and data fetching.