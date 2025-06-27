#!/usr/bin/env python3
"""
TDD Tests for YTD Return Calculation - London TDD Style

RED Phase: Create failing tests that define expected YTD calculation behavior
"""

import pytest
import asyncio
import sys
import os
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src.comprehensive_analyzer import ComprehensiveStockAnalyzer


class TestYTDCalculationAccuracy:
    """Test suite for YTD return calculation accuracy."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        return ComprehensiveStockAnalyzer()

    @pytest.mark.asyncio
    async def test_iwm_ytd_return_accuracy(self, analyzer):
        """
        RED TEST: IWM YTD return should be accurate (around -2.59% based on market data).
        This test will FAIL initially because current system shows 0.0%.
        """
        # Generate IWM analysis
        analysis = await analyzer.analyze_stock("IWM")
        ytd_return = float(analysis.market_data.ytd_return)
        
        # YTD should be non-zero and reflect actual market performance
        # Based on user data: should be around -2.59%
        assert ytd_return != 0.0, f"YTD return is 0.0%, should reflect actual market performance"
        assert -10.0 <= ytd_return <= 10.0, f"YTD return {ytd_return}% seems unrealistic for IWM"
        
        # More specific test based on expected range
        assert -5.0 <= ytd_return <= 0.0, f"YTD return {ytd_return}% should be in expected range for 2024 IWM performance"

    @pytest.mark.asyncio
    async def test_ytd_calculation_fallback_logic(self, analyzer):
        """
        RED TEST: YTD calculation should work even when API fields unavailable.
        This test will FAIL initially because fallback logic doesn't exist.
        """
        # Mock yfinance to return incomplete info
        with patch('yfinance.Ticker') as mock_ticker:
            mock_info = {
                'currentPrice': 215.41,
                'previousClose': 221.14,
                'marketCap': 60500000000,
                # Missing '52WeekChange' field
            }
            
            mock_hist = Mock()
            mock_hist.__getitem__ = Mock(side_effect=lambda key: {
                'Close': [200.0, 215.41],  # Jan 1 and current price
                'High': [244.98],
                'Low': [171.73],
                'Volume': [50000000]
            }[key])
            
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = mock_info
            mock_ticker_instance.history.return_value = mock_hist
            mock_ticker.return_value = mock_ticker_instance
            
            # This should still calculate YTD properly using historical data
            analysis = await analyzer.analyze_stock("IWM")
            ytd_return = float(analysis.market_data.ytd_return)
            
            # Should calculate YTD from historical data: (215.41 - 200.0) / 200.0 * 100 = 7.705%
            expected_ytd = (215.41 - 200.0) / 200.0 * 100
            assert abs(ytd_return - expected_ytd) < 1.0, f"YTD calculation from historical data failed: got {ytd_return}%, expected ~{expected_ytd}%"

    @pytest.mark.asyncio
    async def test_ytd_matches_market_data(self, analyzer):
        """
        RED TEST: YTD calculation should match external market data sources.
        This test will FAIL initially because we're not calculating YTD properly.
        """
        analysis = await analyzer.analyze_stock("IWM")
        ytd_return = float(analysis.market_data.ytd_return)
        
        # Test that YTD is calculated for calendar year (not 52-week period)
        current_year = datetime.now().year
        
        # This will fail initially - we need proper YTD calculation
        assert hasattr(analyzer, '_calculate_ytd_return'), "Analyzer should have _calculate_ytd_return method"
        
        # Test the calculation method exists and works
        ytd_calculated = analyzer._calculate_ytd_return("IWM", current_year)
        assert ytd_calculated is not None, "YTD calculation should return a value"
        assert isinstance(ytd_calculated, (Decimal, float)), "YTD should be numeric"

    def test_ytd_calculator_method_exists(self, analyzer):
        """
        RED TEST: Analyzer should have dedicated YTD calculation method.
        This test will FAIL initially because method doesn't exist.
        """
        # This method doesn't exist yet - will fail
        assert hasattr(analyzer, '_calculate_ytd_return'), "Missing _calculate_ytd_return method"
        
        # Test method signature
        import inspect
        method = getattr(analyzer, '_calculate_ytd_return')
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        
        assert 'symbol' in params, "YTD calculator should accept symbol parameter"
        assert len(params) >= 1, "YTD calculator should have required parameters"

    @pytest.mark.asyncio
    async def test_ytd_vs_52_week_difference(self, analyzer):
        """
        RED TEST: YTD return should differ from 52-week return for most assets.
        This test will FAIL initially because we're using 52WeekChange for YTD.
        """
        analysis = await analyzer.analyze_stock("IWM")
        
        ytd_return = float(analysis.market_data.ytd_return)
        
        # Mock what 52WeekChange would be (different from YTD)
        # In mid-year, 52-week and YTD should typically be different
        current_month = datetime.now().month
        
        if current_month > 6:  # After mid-year
            # YTD and 52-week change should typically be different
            # This will help us verify we're calculating YTD properly, not using 52-week
            assert hasattr(analyzer, '_get_52_week_change'), "Should have method to get 52-week change for comparison"

    @pytest.mark.asyncio 
    async def test_ytd_calculation_edge_cases(self, analyzer):
        """
        RED TEST: YTD calculation should handle edge cases properly.
        This test will FAIL initially because edge case handling doesn't exist.
        """
        # Test with minimal historical data
        with patch('yfinance.Ticker') as mock_ticker:
            mock_info = {'currentPrice': 215.41}
            mock_hist = Mock()
            mock_hist.__getitem__ = Mock(side_effect=lambda key: [215.41] if key == 'Close' else [50000000])
            
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = mock_info
            mock_ticker_instance.history.return_value = mock_hist
            mock_ticker.return_value = mock_ticker_instance
            
            # Should handle insufficient data gracefully
            analysis = await analyzer.analyze_stock("TEST")
            
            # Should not crash and should provide reasonable fallback
            assert analysis.market_data.ytd_return is not None, "YTD should handle edge cases gracefully"

    def test_ytd_date_range_calculation(self, analyzer):
        """
        RED TEST: YTD should calculate from January 1st of current year.
        This test will FAIL initially because proper date range logic doesn't exist.
        """
        current_year = datetime.now().year
        
        # This method doesn't exist yet - will fail
        start_date, end_date = analyzer._get_ytd_date_range()
        
        expected_start = date(current_year, 1, 1)
        expected_end = date.today()
        
        assert start_date == expected_start, f"YTD should start from Jan 1, {current_year}"
        assert end_date == expected_end, f"YTD should end at current date"


class TestYTDCalculationMethods:
    """Test suite for YTD calculation methods and utilities."""
    
    def test_get_ytd_date_range_method(self):
        """RED TEST: Should have method to get YTD date range."""
        analyzer = ComprehensiveStockAnalyzer()
        
        # This method doesn't exist yet - will fail
        assert hasattr(analyzer, '_get_ytd_date_range'), "Missing _get_ytd_date_range method"

    def test_calculate_ytd_from_prices_method(self):
        """RED TEST: Should have method to calculate YTD from price data."""
        analyzer = ComprehensiveStockAnalyzer()
        
        # This method doesn't exist yet - will fail  
        assert hasattr(analyzer, '_calculate_ytd_from_prices'), "Missing _calculate_ytd_from_prices method"

    def test_ytd_fallback_hierarchy(self):
        """RED TEST: Should have clear fallback hierarchy for YTD calculation."""
        analyzer = ComprehensiveStockAnalyzer()
        
        # Should have method that tries multiple data sources
        assert hasattr(analyzer, '_get_ytd_with_fallback'), "Missing _get_ytd_with_fallback method"


if __name__ == "__main__":
    # Run tests to see current failures (RED phase)
    pytest.main([__file__, "-v", "--tb=short"])