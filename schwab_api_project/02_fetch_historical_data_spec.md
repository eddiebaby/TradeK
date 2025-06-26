# Phase 2: Fetch Historical Price Data

**Objective:** Create a pseudocode plan to fetch historical price data for SPY and QQQ for the last month using the Schwab API.

## 1. Project Context & Requirements

- **:Context:** The existing application in [`schwab_api_project/main.py`](schwab_api_project/main.py) handles authentication with the Schwab API.
- **:Problem:** The application needs a new capability to fetch historical price data for specific stock symbols.
- **:Solution:** A new modular function will be created to handle data fetching, which will then be integrated into the main application flow.

## 2. Modular Design

### 2.1. New Module: `data_fetcher.py` (or similar)

A new module should be created to encapsulate data fetching logic, promoting separation of concerns.

#### **`fetch_historical_price_data(client, symbol, start_date, end_date)`**

This function will be responsible for fetching the historical data for a single symbol.

**:ComponentRole:** Data Fetcher

```pseudocode
FUNCTION fetch_historical_price_data(client, symbol, frequency_type, frequency, start_date, end_date):
  // :Context: This function uses the authenticated Schwab API client to retrieve price history.
  // :Pattern: API Wrapper. Encapsulates the logic for a specific API call.

  // 1. Log the attempt to fetch data for the given symbol.
  PRINT "Fetching historical data for symbol:", symbol

  // 2. Make the API call to get price history.
  //    The schwab-py library provides a `get_price_history` method.
  //    We need to determine the correct parameters based on the library's documentation.
  //    Let's assume the parameters are: symbol, period_type, period, frequency_type, frequency, start_date, end_date.
  //    For this task, we'll use daily frequency over the last month.
  response = client.get_price_history(
      symbol=symbol,
      frequency_type=frequency_type,
      frequency=frequency,
      start_date=start_date,
      end_date=end_date
  )

  // 3. Handle the API response.
  IF response.ok THEN
    // 3.1. On success, parse the JSON data.
    data = response.json()
    PRINT "Successfully fetched data for", symbol
    RETURN data['candles']
  ELSE
    // 3.2. On failure, log the error and return None.
    PRINT "Failed to fetch data for", symbol
    PRINT "Status Code:", response.status_code
    PRINT "Response:", response.text
    RETURN None
  END IF
END FUNCTION
```

### 2.2. Modifications to `main.py`

The main function will be updated to use the new data fetching capability.

#### **`main()`**

**:ComponentRole:** Application Orchestrator

```pseudocode
// Import necessary modules, including the new data_fetcher
IMPORT data_fetcher
IMPORT datetime

FUNCTION main():
  // ... (Existing authentication logic remains the same)
  // 1. Authenticate and get the client object.
  client = authenticate_with_schwab() // Assumes authentication logic is refactored for clarity

  IF client is not None THEN
    // 2. Fetch account numbers (existing functionality).
    fetch_and_print_account_numbers(client)

    // 3. Define symbols and time frame for historical data.
    // :Configuration: Symbols are defined here but could be externalized to config.py.
    symbols_to_fetch = ["SPY", "QQQ"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    // 4. Loop through symbols and fetch data.
    FOR each symbol in symbols_to_fetch:
      // :Context: Call the new function to get data for each symbol.
      historical_data = data_fetcher.fetch_historical_price_data(
          client=client,
          symbol=symbol,
          frequency_type='DAILY',
          frequency=1,
          start_date=start_date,
          end_date=end_date
      )

      // 5. Process and display the fetched data.
      IF historical_data is not None THEN
        PRINT "----------------------------------------"
        PRINT "Historical Data for", symbol
        // :Pattern: Data Formatting. Present the data in a user-friendly way.
        FOR each candle in historical_data:
          // Example: "2025-05-07: Open=500.00, High=505.50, Low=499.00, Close=505.00, Volume=1234567"
          PRINT f"{datetime.fromtimestamp(candle['datetime'] / 1000).strftime('%Y-%m-%d')}: Open={candle['open']}, High={candle['high']}, Low={candle['low']}, Close={candle['close']}, Volume={candle['volume']}"
        END FOR
        PRINT "----------------------------------------"
      END IF
    END FOR
  END IF
END FUNCTION
```

## 3. TDD Anchors

These are the tests that should be created in the `tests/` directory to validate the new functionality.

### `test_data_fetcher.py`

-   **`test_fetch_historical_data_success`**:
    -   **Given**: A mocked `client` object and a valid stock `symbol` ("SPY").
    -   **When**: `fetch_historical_price_data` is called.
    -   **Then**: The function should return a list of candle data structures.
    -   **And**: The mocked `client.get_price_history` method is called with the correct parameters.

-   **`test_fetch_historical_data_invalid_symbol`**:
    -   **Given**: A mocked `client` that returns a failure response (e.g., 404 Not Found) for an invalid `symbol` ("INVALID").
    -   **When**: `fetch_historical_price_data` is called with the invalid symbol.
    -   **Then**: The function should return `None`.
    -   **And**: An appropriate error message is logged to the console.

-   **`test_fetch_historical_data_api_error`**:
    -   **Given**: A mocked `client` that raises an exception or returns a 500 server error.
    -   **When**: `fetch_historical_price_data` is called.
    -   **Then**: The function should handle the error gracefully and return `None`.

### `test_main.py`

-   **`test_main_flow_with_data_fetching`**:
    -   **Given**: A mocked authentication flow and a mocked `fetch_historical_price_data` function.
    -   **When**: `main()` is executed.
    -   **Then**: The `main` function should call the `fetch_historical_price_data` function for each symbol ("SPY", "QQQ").
    -   **And**: The formatted output for the historical data is printed to the console.

## 4. Edge Cases & Constraints

-   **API Rate Limiting**: The Schwab API may have rate limits. The implementation should be mindful of this, although for two symbols, it's unlikely to be an issue.
-   **Market Holidays/Weekends**: The API might not return data for non-trading days. The output formatting should be robust to handle this.
-   **Timezone Handling**: All date/time operations should consistently use the same timezone, preferably UTC, to avoid ambiguity. The API returns timestamps in milliseconds since the epoch.