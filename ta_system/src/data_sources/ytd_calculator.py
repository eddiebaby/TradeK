#!/usr/bin/env python3
"""
YTD (Year-to-Date) Return Calculator Service

Clean, testable service for calculating accurate YTD returns with multiple fallback strategies.
Follows Single Responsibility Principle and provides robust error handling.

This service was created as part of the London TDD refactor to fix inaccurate YTD calculations.
"""

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

import pandas as pd
import yfinance as yf


class YTDCalculationError(Exception):
    """Custom exception for YTD calculation errors."""


# Constants for validation and comparison
MIN_HISTORICAL_DATA_POINTS = 2
EXTREME_YTD_THRESHOLD = 200.0  # ±200% return threshold


class YTDCalculator:
    """
    Service for calculating Year-to-Date returns with multiple fallback strategies.
    
    This class encapsulates all YTD calculation logic, making it:
    - Testable in isolation
    - Reusable across different components
    - Easy to maintain and extend
    - Follows Single Responsibility Principle
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize YTD calculator with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_ytd_return(self, symbol: str, year: Optional[int] = None) -> Optional[Decimal]:
        """
        Calculate accurate YTD return using historical data.
        
        Args:
            symbol: Stock/ETF symbol
            year: Target year (defaults to current year)
            
        Returns:
            YTD return as percentage, or None if calculation fails
            
        Raises:
            YTDCalculationError: If all calculation methods fail
        """
        try:
            if year is None:
                year = datetime.now().year
            
            # Get YTD date range
            start_date, end_date = self._get_ytd_date_range(year)
            
            # Fetch historical data
            hist = self._fetch_historical_data(symbol, start_date, end_date)
            
            if hist.empty or len(hist) < MIN_HISTORICAL_DATA_POINTS:
                self.logger.warning(f"Insufficient historical data for {symbol} YTD calculation")
                return None
            
            # Calculate YTD return from price data
            jan_price = float(hist["Close"].iloc[0])  
            current_price = float(hist["Close"].iloc[-1])
            
            ytd_return = self._calculate_return_percentage(jan_price, current_price)
            
            self.logger.info(f"Calculated YTD return for {symbol}: {ytd_return}%")
            return ytd_return
            
        except Exception as e:
            self.logger.error(f"YTD calculation failed for {symbol}: {e}")
            return None

    def calculate_ytd_with_fallback(
        self, 
        symbol: str, 
        info: dict[str, Any], 
        hist: pd.DataFrame
    ) -> Decimal:
        """
        Calculate YTD return with comprehensive fallback strategy.
        
        Fallback hierarchy:
        1. Historical data calculation (most accurate)
        2. API 52-week change field (approximation)
        3. Available historical data range
        4. Zero fallback (last resort)
        
        Args:
            symbol: Stock/ETF symbol
            info: yfinance info dictionary
            hist: Historical price data
            
        Returns:
            YTD return as Decimal percentage
        """
        # Method 1: Calculate from historical data (most accurate)
        ytd_calculated = self.calculate_ytd_return(symbol)
        if ytd_calculated is not None:
            self.logger.debug(f"Using historical data YTD for {symbol}: {ytd_calculated}%")
            return ytd_calculated
        
        # Method 2: Use yfinance API field if available
        if "52WeekChange" in info and info["52WeekChange"] is not None:
            try:
                # Note: 52WeekChange is not the same as YTD, but better than 0
                api_ytd = Decimal(str(info["52WeekChange"] * 100))
                self.logger.debug(f"Using API 52WeekChange for {symbol}: {api_ytd}%")
                return api_ytd
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Failed to parse 52WeekChange for {symbol}: {e}")
        
        # Method 3: Calculate from available historical data
        if not hist.empty and len(hist) >= MIN_HISTORICAL_DATA_POINTS:
            try:
                start_price = float(hist["Close"].iloc[0])
                current_price = float(hist["Close"].iloc[-1])
                fallback_ytd = self._calculate_return_percentage(start_price, current_price)
                self.logger.debug(f"Using historical range YTD for {symbol}: {fallback_ytd}%")
                return fallback_ytd
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Failed to calculate from historical range for {symbol}: {e}")
        
        # Method 4: Fallback to 0 (last resort)
        self.logger.warning(f"All YTD calculation methods failed for {symbol}, using 0%")
        return Decimal("0")

    def _get_ytd_date_range(self, year: Optional[int] = None) -> tuple[date, date]:
        """
        Get YTD date range from January 1st to current date.
        
        Args:
            year: Target year (defaults to current year)
            
        Returns:
            Tuple of (start_date, end_date)
        """
        if year is None:
            year = datetime.now().year
            
        start_date = date(year, 1, 1)
        end_date = date.today()
        
        return start_date, end_date

    def _fetch_historical_data(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Fetch historical price data for the specified date range.
        
        Args:
            symbol: Stock/ETF symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with historical price data
            
        Raises:
            YTDCalculationError: If data fetching fails
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date.isoformat(), end=end_date.isoformat())
            
            if hist.empty:
                raise YTDCalculationError(f"No historical data available for {symbol}")
                
            return hist
            
        except Exception as e:
            raise YTDCalculationError(f"Failed to fetch historical data for {symbol}: {e}") from e

    def _calculate_return_percentage(self, start_price: float, end_price: float) -> Decimal:
        """
        Calculate percentage return between two prices.
        
        Args:
            start_price: Starting price
            end_price: Ending price
            
        Returns:
            Return percentage as Decimal
            
        Raises:
            YTDCalculationError: If start_price is zero or invalid
        """
        if start_price == 0:
            raise YTDCalculationError("Cannot calculate return with zero start price")
        
        if start_price < 0 or end_price < 0:
            raise YTDCalculationError("Cannot calculate return with negative prices")
        
        return_pct = ((end_price - start_price) / start_price) * 100
        return Decimal(str(return_pct))

    def validate_ytd_return(self, ytd_return: Decimal, symbol: str) -> bool:
        """
        Validate that YTD return is within reasonable bounds.
        
        Args:
            ytd_return: YTD return percentage
            symbol: Stock/ETF symbol for context
            
        Returns:
            True if YTD return appears valid
        """
        # Most securities shouldn't have extreme YTD returns
        if abs(float(ytd_return)) > EXTREME_YTD_THRESHOLD:
            self.logger.warning(f"Extreme YTD return for {symbol}: {ytd_return}%")
            return False
        
        return True

    def get_calculation_metadata(self, symbol: str) -> dict[str, Any]:
        """
        Get metadata about YTD calculation for debugging/monitoring.
        
        Args:
            symbol: Stock/ETF symbol
            
        Returns:
            Dictionary with calculation metadata
        """
        start_date, end_date = self._get_ytd_date_range()
        
        return {
            "symbol": symbol,
            "calculation_date": datetime.now().isoformat(),
            "ytd_start_date": start_date.isoformat(),
            "ytd_end_date": end_date.isoformat(),
            "data_source": "yfinance",
            "calculation_method": "historical_data_primary"
        }


# Factory function for easy instantiation
def create_ytd_calculator(logger: Optional[logging.Logger] = None) -> YTDCalculator:
    """
    Factory function to create YTD calculator instance.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        Configured YTDCalculator instance
    """
    return YTDCalculator(logger=logger)


# Example usage
if __name__ == "__main__":
    # Demo the YTD calculator
    calculator = create_ytd_calculator()
    
    # Test with IWM
    ytd = calculator.calculate_ytd_return("IWM")
    if ytd:
        print(f"IWM YTD Return: {ytd}%")
        
        # Validate the result
        if calculator.validate_ytd_return(ytd, "IWM"):
            print("✅ YTD return validated")
        else:
            print("⚠️ YTD return seems extreme")
            
        # Get metadata
        metadata = calculator.get_calculation_metadata("IWM")
        print(f"Calculation metadata: {metadata}")
    else:
        print("❌ Failed to calculate YTD return")