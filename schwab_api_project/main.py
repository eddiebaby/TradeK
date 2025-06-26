import importlib
import schwab_api_project.config as config
from schwab.auth import client_from_token_file
from schwab.client import Client
from pathlib import Path
import webbrowser
import json
from datetime import datetime, timedelta # Added

import logging

# :ComponentRole: Data Fetcher
# :Context: This function uses the authenticated Schwab API client to retrieve price history.
# :Pattern: API Wrapper. Encapsulates the logic for a specific API call.
def fetch_historical_price_data(client, symbol, frequency_type, frequency, start_date, end_date):
    """
    Fetches historical price data for a given symbol.
    """
    print(f"Fetching historical data for symbol: {symbol}, FreqType: {frequency_type}, Freq: {frequency}, Start: {start_date.strftime('%Y-%m-%d')}, End: {end_date.strftime('%Y-%m-%d')}")
    try:
        response = client.get_price_history(
            symbol=symbol,
            frequency_type=frequency_type,
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
        )

        if response.ok:
            data = response.json()
            if data.get('empty') == True or not data.get('candles'):
                 print(f"No candles data found for {symbol} in the response.")
                 return [] # Return empty list if no candles or explicitly empty
            print(f"Successfully fetched data for {symbol}")
            return data['candles']
        else:
            print(f"Failed to fetch data for {symbol}")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"An error occurred while fetching price history for {symbol}: {e}")
        return None

def main():
    """Main function to perform authentication, fetch account numbers, and fetch historical price data."""
    # Configure logging to show debug messages from the schwab library
    logging.basicConfig(level=logging.INFO) # Set root logger
    schwab_logger = logging.getLogger('schwab')
    schwab_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    schwab_logger.addHandler(handler)
    schwab_logger.propagate = False # Prevent duplicate logging messages
    print("Starting Schwab API connection process...")
    try:
        # 1. Authenticate and get the client object.
        print("Authenticating with Schwab...")
        # :Solution: Use an absolute path for the token file and add a debug function.
        token_path = Path(__file__).parent / 'token.json'
        print(f"Token path set to: {token_path}")

        def write_token(token_data):
            print(f"Received token data: {token_data}")
            with open(token_path, 'w') as f:
                json.dump(token_data, f)
            print("Token successfully written to file.")

        client = client_from_token_file(
            token_path,
            config.api_key,
            config.app_secret
        )

        print("Authentication successful.")

        # 2. Fetch account numbers (existing functionality).
        print("\nFetching account numbers...")
        response_acc = client.get_account_numbers()
        if response_acc.ok:
            account_numbers = response_acc.json()
            print("Successfully fetched account numbers:")
            print(json.dumps(account_numbers, indent=4))
        else:
            print(f"Failed to fetch account numbers: {response_acc.status_code} - {response_acc.text}")

        # 3. Define symbols and time frame for historical data.
        # :Configuration: Symbols are defined here but could be externalized to config.py.
        symbols_to_fetch = ["SPY", "QQQ"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        print(f"\nFetching historical data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} for symbols: {', '.join(symbols_to_fetch)}")

        # 4. Loop through symbols and fetch data.
        for symbol_item in symbols_to_fetch:
            # :Context: Call the new function to get data for each symbol.
            historical_data = fetch_historical_price_data(
                client=client,
                symbol=symbol_item,
                frequency_type=Client.PriceHistory.FrequencyType.DAILY, # As per spec example
                frequency=Client.PriceHistory.Frequency.DAILY,          # As per spec example
                start_date=start_date,
                end_date=end_date
            )

            # 5. Process and display the fetched data.
            if historical_data: # Check if historical_data is not None and not empty
                print("----------------------------------------")
                print(f"Historical Data for {symbol_item}")
                # :Pattern: Data Formatting. Present the data in a user-friendly way.
                for candle in historical_data:
                    dt_object = datetime.fromtimestamp(candle['datetime'] / 1000)
                    print(f"{dt_object.strftime('%Y-%m-%d')}: Open={candle.get('open', 'N/A')}, High={candle.get('high', 'N/A')}, Low={candle.get('low', 'N/A')}, Close={candle.get('close', 'N/A')}, Volume={candle.get('volume', 'N/A')}")
                print("----------------------------------------")
            elif historical_data is None: # Explicitly None means an error occurred during fetch
                print(f"Error fetching historical data for {symbol_item}.")
            else: # Empty list means no data found for the period
                print(f"No historical data found for {symbol_item} for the specified period.")
    
    except Exception as e:
        print(f"An critical error occurred in main: {e}")

if __name__ == "__main__":
    main()