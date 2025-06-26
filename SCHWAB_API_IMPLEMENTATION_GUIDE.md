# Schwab API Integration Implementation Guide

## Overview
This guide provides comprehensive instructions for implementing Charles Schwab API integration in the LDES (Liquidity Detection & Execution System). The implementation supports both real-time streaming data and historical data collection with production-ready OAuth 2.0 authentication.

## Table of Contents
1. [Authentication Setup](#authentication-setup)
2. [Installation & Dependencies](#installation--dependencies)
3. [Configuration](#configuration)
4. [OAuth 2.0 Implementation](#oauth-20-implementation)
5. [Market Data Collection](#market-data-collection)
6. [Streaming Data](#streaming-data)
7. [Rate Limiting & Error Handling](#rate-limiting--error-handling)
8. [Production Deployment](#production-deployment)
9. [Testing & Validation](#testing--validation)
10. [Integration with LDES](#integration-with-ldes)

## Authentication Setup

### Prerequisites
1. **Schwab Developer Account**: Create an account at [developer.schwab.com](https://developer.schwab.com)
2. **Application Registration**: Register your application to obtain:
   - `APP_KEY` (Consumer Key)
   - `APP_SECRET` (Consumer Secret)
3. **Callback URL Configuration**: Set callback URL to `https://127.0.0.1:8000/callback`

### Application Approval Process
- **Timeline**: 3-7 business days
- **Requirements**: Detailed application description, intended use case
- **Important**: Test in sandbox environment first

## Installation & Dependencies

### Required Python Packages
```bash
pip install schwab-py
pip install aiohttp
pip install websockets
```

### Add to requirements.txt
```txt
schwab-py>=1.5.0
aiohttp>=3.8.0
websockets>=11.0.0
```

## Configuration

### Environment Variables
```bash
# Schwab API Configuration
export SCHWAB_APP_KEY="your_app_key_here"
export SCHWAB_SECRET="your_app_secret_here" 
export SCHWAB_ACCOUNT_ID="your_account_id"
export SCHWAB_REDIRECT_URI="https://127.0.0.1:8000/callback"

# Token Management
export SCHWAB_TOKEN_DIR="/secure/path/to/tokens"
export SCHWAB_TOKEN_FILE="schwab_token.json"

# Server Mode (for production)
export SCHWAB_SERVER_MODE="true"
```

### Configuration Updates
The `MarketDataConfig` class already includes Schwab configuration:

```python
class MarketDataConfig(BaseModel):
    # Schwab configuration
    schwab_app_key: Optional[str] = Field(default_factory=lambda: os.getenv("SCHWAB_APP_KEY"))
    schwab_secret: Optional[str] = Field(default_factory=lambda: os.getenv("SCHWAB_SECRET"))
    schwab_account_id: Optional[str] = Field(default_factory=lambda: os.getenv("SCHWAB_ACCOUNT_ID"))
    schwab_redirect_uri: str = Field(default="http://localhost:8000/callback")
```

## OAuth 2.0 Implementation

### Authentication Flow

#### 1. Interactive Mode (Development)
```python
from schwab import auth

# Easy client handles OAuth flow automatically
client = auth.easy_client(
    api_key=app_key,
    app_secret=app_secret,
    callback_url=redirect_uri,
    token_path=token_path
)
```

#### 2. Server Mode (Production)
```python
# For production servers without GUI
if os.getenv("SCHWAB_SERVER_MODE", "false").lower() == "true":
    if os.path.exists(token_path):
        client = auth.client_from_token_file(
            token_path, app_key, app_secret
        )
    else:
        raise RuntimeError("Server mode requires existing token file")
```

### Token Management Strategy

#### Token Lifecycle
- **Access Token**: Valid for 30 minutes
- **Refresh Token**: Valid for 7 days
- **Auto-refresh**: Handled automatically by schwab-py

#### Secure Token Storage
```python
def _get_token_path(self) -> str:
    """Get secure token storage path."""
    token_dir = os.getenv("SCHWAB_TOKEN_DIR", tempfile.gettempdir())
    token_file = os.getenv("SCHWAB_TOKEN_FILE", "schwab_token.json")
    return os.path.join(token_dir, token_file)
```

#### Production Token Setup
1. **Initial Setup**: Run authentication in interactive mode once
2. **Token File**: Securely transfer token file to production server
3. **File Permissions**: Set strict permissions (600) on token file
4. **Backup Strategy**: Implement token backup and rotation

## Market Data Collection

### Supported Data Types

#### 1. Real-time Streaming Data
- **Level 1 Quotes**: Bid/ask prices and sizes
- **Level 2 Order Books**: 10-level market depth
- **Chart Data**: OHLCV minute bars
- **Account Activity**: Position updates

#### 2. Historical Data
- **Minute Data**: Up to 48 days
- **5/10/15/30 Minute**: ~9 months
- **Daily/Weekly**: Back to 1985

### API Methods

#### Historical Data Collection
```python
async def get_historical_data(
    self,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = "1min"
) -> List[MarketData]:
    """Get historical market data with rate limiting."""
    
    await self._enforce_rate_limit()
    
    # Select appropriate method based on timeframe
    if timeframe == "1min":
        method = self.http_client.get_price_history_every_minute
    elif timeframe == "1day":
        method = self.http_client.get_price_history_every_day
    # ... other timeframes
    
    response = await asyncio.get_event_loop().run_in_executor(
        None, 
        lambda: method(symbol, start_date=start_date, end_date=end_date)
    )
    
    return self._convert_price_history_to_market_data(symbol, response.json())
```

#### Real-time Quotes
```python
async def get_latest_quote(self, symbol: str) -> Optional[MarketData]:
    """Get latest quote with rate limiting."""
    await self._enforce_rate_limit()
    
    response = await asyncio.get_event_loop().run_in_executor(
        None, self.http_client.get_quote, symbol
    )
    
    return self._convert_quote_to_market_data(symbol, response.json())
```

### Data Normalization

#### Converting Schwab Data to LDES Format
```python
def _convert_level1_to_market_data(self, quote_data: Dict[str, Any]) -> Optional[MarketData]:
    """Convert Schwab Level 1 quote to normalized MarketData."""
    return MarketData(
        symbol=quote_data.get('key', ''),
        timestamp=datetime.now(),
        bid_price=Decimal(str(quote_data.get('BID_PRICE'))) if quote_data.get('BID_PRICE') else None,
        ask_price=Decimal(str(quote_data.get('ASK_PRICE'))) if quote_data.get('ASK_PRICE') else None,
        last_price=Decimal(str(quote_data.get('LAST_PRICE'))) if quote_data.get('LAST_PRICE') else None,
        bid_size=int(quote_data.get('BID_SIZE')) if quote_data.get('BID_SIZE') else None,
        ask_size=int(quote_data.get('ASK_SIZE')) if quote_data.get('ASK_SIZE') else None,
        volume=int(quote_data.get('TOTAL_VOLUME')) if quote_data.get('TOTAL_VOLUME') else None,
        source="schwab"
    )
```

## Streaming Data

### WebSocket Connection Management

#### Connection Setup
```python
async def _initialize_stream_client(self) -> None:
    """Initialize streaming client with authentication."""
    self.stream_client = StreamClient(self.http_client, account_id=None)
    self._setup_stream_handlers()
    
    # Login to streaming
    await self.stream_client.login()
    self._stream_connected = True
```

#### Subscription Management
```python
async def subscribe(self, symbols: List[str]) -> None:
    """Subscribe to multiple data types."""
    # Level 1 quotes
    await self.stream_client.level_one_equity_subs(symbols)
    
    # Level 2 order books
    await self.stream_client.nasdaq_book_subs(symbols)
    await self.stream_client.nyse_book_subs(symbols)
    
    # Chart data
    await self.stream_client.chart_equity_subs(symbols)
```

#### Message Handling
```python
async def _handle_stream_messages(self) -> None:
    """Handle incoming streaming messages."""
    while self._stream_connected:
        await self.stream_client.handle_message()
        await asyncio.sleep(0.001)  # Prevent busy loop
```

### Stream Data Processing

#### Level 1 Handler
```python
async def handle_level1_quotes(message: Dict[str, Any]):
    """Process Level 1 quote updates."""
    if 'content' in message:
        for quote_data in message['content']:
            market_data = self._convert_level1_to_market_data(quote_data)
            if market_data:
                await self._data_queue.put(market_data)
                self.quotes_received += 1
```

#### Level 2 Handler
```python
async def handle_level2_books(message: Dict[str, Any]):
    """Process Level 2 order book updates."""
    if 'content' in message:
        for book_data in message['content']:
            market_data = self._convert_level2_to_market_data(book_data)
            if market_data:
                await self._data_queue.put(market_data)
                self.level2_updates_received += 1
```

## Rate Limiting & Error Handling

### Rate Limiting Implementation

#### Conservative Rate Limits
```python
# Rate limiting configuration
self._rate_limit_window = 60  # seconds
self._max_requests_per_minute = 120  # Conservative estimate
```

#### Rate Limit Enforcement
```python
async def _enforce_rate_limit(self) -> None:
    """Enforce API rate limits."""
    current_time = asyncio.get_event_loop().time()
    
    # Reset counter if window has passed
    if current_time - self._last_request_time > self._rate_limit_window:
        self._request_count = 0
        self._last_request_time = current_time
    
    # Check if approaching rate limit
    if self._request_count >= self._max_requests_per_minute:
        wait_time = self._rate_limit_window - (current_time - self._last_request_time)
        if wait_time > 0:
            logger.warning(f"Rate limit reached, waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
            self._request_count = 0
```

### Error Handling Strategy

#### Connection Resilience
```python
async def _connect_with_retry(self, max_retries: int = 3) -> None:
    """Connect with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            await self._initialize_http_client()
            await self._initialize_stream_client()
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                raise
```

#### Token Refresh Handling
```python
async def _handle_token_refresh(self) -> None:
    """Handle token refresh failures."""
    try:
        # Attempt to refresh token
        if self.http_client:
            # schwab-py handles this automatically
            pass
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        # Trigger re-authentication
        await self._reauthenticate()
```

## Production Deployment

### Security Best Practices

#### 1. Token Security
- Store tokens in encrypted filesystem
- Use environment variables for credentials
- Implement token rotation
- Monitor token expiration

#### 2. Network Security
- Use HTTPS for all communications
- Implement certificate pinning
- Monitor for man-in-the-middle attacks

#### 3. Access Control
- Principle of least privilege
- Regular credential rotation
- Audit logging
- IP whitelisting if possible

### Monitoring & Logging

#### Metrics Collection
```python
def get_provider_info(self) -> Dict[str, Any]:
    """Get comprehensive provider metrics."""
    return {
        "name": "Charles Schwab",
        "is_connected": self.is_connected,
        "stream_connected": self._stream_connected,
        "quotes_received": self.quotes_received,
        "level2_updates_received": self.level2_updates_received,
        "errors_count": self.errors_count,
        "rate_limit_remaining": max(0, self._max_requests_per_minute - self._request_count),
        "token_expires_in": self._get_token_expiry_time()
    }
```

#### Health Checks
```python
async def health_check(self) -> Dict[str, Any]:
    """Comprehensive health check."""
    try:
        # Test HTTP connection
        test_quote = await self.get_latest_quote("SPY")
        
        # Test streaming connection
        stream_healthy = self._stream_connected and self.stream_client is not None
        
        return {
            "status": "healthy" if test_quote and stream_healthy else "degraded",
            "http_client": test_quote is not None,
            "stream_client": stream_healthy,
            "last_data_time": self.last_data_timestamp,
            "error_rate": self.errors_count / max(1, self.quotes_received)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

### Deployment Architecture

#### Container Configuration
```dockerfile
# Add to Dockerfile
RUN pip install schwab-py

# Environment variables
ENV SCHWAB_SERVER_MODE=true
ENV SCHWAB_TOKEN_DIR=/app/data/tokens
```

#### Kubernetes ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: schwab-config
data:
  SCHWAB_REDIRECT_URI: "https://127.0.0.1:8000/callback"
  SCHWAB_SERVER_MODE: "true"
```

## Testing & Validation

### Unit Tests
```python
import pytest
from unittest.mock import Mock, AsyncMock

class TestSchwabDataProvider:
    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.schwab_app_key = "test_key"
        config.schwab_secret = "test_secret"
        config.schwab_redirect_uri = "https://127.0.0.1:8000/callback"
        return config
    
    @pytest.mark.asyncio
    async def test_connection(self, mock_config):
        provider = SchwabDataProvider(mock_config)
        # Mock the HTTP client initialization
        provider._initialize_http_client = AsyncMock()
        provider._initialize_stream_client = AsyncMock()
        
        await provider.connect()
        
        assert provider.is_connected
        assert provider._initialize_http_client.called
        assert provider._initialize_stream_client.called
```

### Integration Tests
```python
@pytest.mark.integration
class TestSchwabIntegration:
    @pytest.mark.asyncio
    async def test_real_data_collection(self):
        """Test against real Schwab API (requires valid credentials)."""
        config = MarketDataConfig()
        provider = SchwabDataProvider(config)
        
        try:
            await provider.connect()
            quote = await provider.get_latest_quote("SPY")
            
            assert quote is not None
            assert quote.symbol == "SPY"
            assert quote.last_price > 0
            
        finally:
            await provider.disconnect()
```

### Mock Provider Testing
```python
@pytest.mark.asyncio
async def test_mock_provider():
    """Test mock provider for development environments."""
    config = MarketDataConfig()
    provider = MockSchwabDataProvider(config)
    
    await provider.connect()
    
    quote = await provider.get_latest_quote("SPY")
    assert quote is not None
    assert quote.symbol == "SPY"
    
    # Test streaming
    stream_count = 0
    async for data in provider.get_stream():
        stream_count += 1
        if stream_count >= 5:  # Test 5 data points
            break
    
    assert stream_count == 5
```

## Integration with LDES

### Adding Schwab Provider to Market Data Collector

#### Update Factory Method
```python
# In market_data_collector.py
from .schwab_client import create_schwab_provider

def create_data_providers(config: LDESConfig) -> Dict[str, MarketDataProvider]:
    """Create all configured data providers."""
    providers = {}
    
    # Existing providers...
    
    # Add Schwab provider
    if config.market_data.schwab_app_key:
        schwab_provider = create_schwab_provider(
            config.market_data, 
            use_mock=not config.is_production()
        )
        providers["schwab"] = schwab_provider
    
    return providers
```

#### Configuration Integration
```python
# Update symbols list
def get_all_symbols(config: LDESConfig) -> List[str]:
    """Get all symbols from all providers."""
    symbols = set()
    
    # Existing symbols...
    
    # Add Schwab symbols
    if config.market_data.schwab_app_key:
        symbols.update([
            "SPY", "QQQ", "IWM", "DIA", "VTI",
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"
        ])
    
    return sorted(list(symbols))
```

### Usage Example
```python
async def main():
    """Example LDES usage with Schwab provider."""
    # Load configuration
    config = LDESConfig()
    
    # Create market data collector
    collector = MarketDataCollector(config)
    
    # Add Schwab provider
    schwab_provider = create_schwab_provider(config.market_data)
    collector.add_provider("schwab", schwab_provider)
    
    # Connect and start collection
    symbols = ["SPY", "QQQ", "AAPL", "MSFT"]
    
    async with collector.managed_collection(symbols) as active_collector:
        # Process real-time data
        async for data in active_collector.get_stream():
            logger.info(f"Received data: {data.symbol} @ {data.last_price}")
```

## Troubleshooting

### Common Issues

#### 1. Authentication Failures
```
Error: invalid_client
Solution: Check token expiration (7-day limit), regenerate if needed
```

#### 2. Rate Limiting
```
Error: Too Many Requests
Solution: Implement exponential backoff, reduce request frequency
```

#### 3. Streaming Disconnections
```
Error: WebSocket connection closed
Solution: Implement reconnection logic with subscription restoration
```

#### 4. Token File Permissions
```
Error: Permission denied
Solution: Set correct file permissions (600) and ownership
```

### Debug Mode
```python
# Enable debug logging
logging.getLogger("schwab").setLevel(logging.DEBUG)
```

### Support Resources
- **Documentation**: https://schwab-py.readthedocs.io/
- **GitHub Issues**: https://github.com/alexgolec/schwab-py/issues
- **Community Discord**: Available through documentation
- **Schwab Developer Support**: developer.schwab.com

## Conclusion

This implementation provides a production-ready Schwab API integration with:

✅ **Complete OAuth 2.0 Flow**: Secure authentication with automatic token refresh  
✅ **Real-time Streaming**: WebSocket-based Level 1 & Level 2 market data  
✅ **Historical Data**: Comprehensive price history access  
✅ **Rate Limiting**: Compliant request throttling  
✅ **Error Handling**: Robust connection management and retry logic  
✅ **Production Ready**: Security best practices and monitoring  
✅ **LDES Integration**: Seamless integration with existing architecture  

The implementation follows established patterns from the existing Alpaca and Binance providers while handling Schwab's specific requirements for OAuth 2.0 authentication and streaming data management.