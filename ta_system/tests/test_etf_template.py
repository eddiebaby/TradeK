#!/usr/bin/env python3
"""
TDD Tests for ETF Template Validation - London TDD Style

RED Phase: Create failing tests that define expected behavior
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src.comprehensive_analyzer import ComprehensiveStockAnalyzer
from src.reports.generator import ReportGenerator


class TestETFTemplateValidation:
    """Test suite for ETF-specific template validation."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        return ComprehensiveStockAnalyzer()
    
    @pytest.fixture 
    def report_generator(self):
        """Create report generator for testing."""
        return ReportGenerator()

    @pytest.mark.asyncio
    async def test_iwm_analysis_has_no_na_values(self, analyzer):
        """
        RED TEST: IWM analysis should contain zero 'N/A' values.
        This test will FAIL initially because current system produces N/A values.
        """
        # Generate IWM analysis
        analysis = await analyzer.analyze_stock("IWM")
        report = await analyzer.generate_report(analysis)
        
        # Count N/A occurrences
        na_count = report.count("N/A")
        
        # This should be zero - will FAIL initially (RED phase)
        assert na_count == 0, f"Found {na_count} 'N/A' values in IWM report. ETF reports should have zero N/A values."

    @pytest.mark.asyncio 
    async def test_etf_template_variables_populated(self, analyzer):
        """
        RED TEST: ETF template should have 95%+ variable population rate.
        This test will FAIL initially due to inappropriate stock template usage.
        """
        analysis = await analyzer.analyze_stock("IWM")
        report = await analyzer.generate_report(analysis)
        
        # Count total template variables vs populated ones
        import re
        remaining_placeholders = len(re.findall(r'\{[A-Z_0-9]+\}', report))
        
        # Should have zero remaining placeholders for ETF
        assert remaining_placeholders == 0, f"Found {remaining_placeholders} unpopulated template variables in ETF report"

    @pytest.mark.asyncio
    async def test_etf_specific_sections_present(self, analyzer):
        """
        RED TEST: ETF analysis should contain ETF-specific sections.
        This test will FAIL initially because we're using stock template.
        """
        analysis = await analyzer.analyze_stock("IWM")
        report = await analyzer.generate_report(analysis)
        
        # ETF reports should contain these sections
        required_etf_sections = [
            "ETF Mechanics & Efficiency",
            "Small-Cap Investment Framework", 
            "Portfolio Implementation Strategies",
            "Expense Ratio",
            "Tracking Error"
        ]
        
        missing_sections = []
        for section in required_etf_sections:
            if section not in report:
                missing_sections.append(section)
        
        assert len(missing_sections) == 0, f"Missing ETF-specific sections: {missing_sections}"

    @pytest.mark.asyncio
    async def test_inappropriate_stock_sections_absent(self, analyzer):
        """
        RED TEST: ETF analysis should NOT contain stock-specific sections.
        This test will FAIL initially because we're using stock template.
        """
        analysis = await analyzer.analyze_stock("IWM")
        report = await analyzer.generate_report(analysis)
        
        # ETF reports should NOT contain these stock-specific sections
        inappropriate_sections = [
            "SEGMENT_1_NAME Revenue",
            "SEGMENT_2_NAME Revenue", 
            "SEGMENT_3_NAME Revenue",
            "STRATEGIC_THEME_1",
            "COMPETITIVE ADVANTAGES"
        ]
        
        found_inappropriate = []
        for section in inappropriate_sections:
            if section in report:
                found_inappropriate.append(section)
        
        assert len(found_inappropriate) == 0, f"Found inappropriate stock sections in ETF report: {found_inappropriate}"

    def test_asset_type_detection(self, analyzer):
        """
        RED TEST: System should detect IWM as ETF, not stock.
        This test will FAIL initially because detection logic doesn't exist.
        """
        # This method doesn't exist yet - will fail
        asset_type = analyzer.detect_asset_type("IWM")
        assert asset_type == "ETF", f"IWM should be detected as ETF, got {asset_type}"

    def test_template_selection_strategy(self, report_generator):
        """
        RED TEST: Report generator should select ETF template for ETF assets.
        This test will FAIL initially because template selection doesn't exist.
        """
        # Mock analysis object
        mock_analysis = Mock()
        mock_analysis.company_profile.symbol = "IWM"
        
        # This method doesn't exist yet - will fail
        selected_template = report_generator.select_template(mock_analysis)
        assert selected_template == "etf_analysis_template.md", f"Should select ETF template for IWM, got {selected_template}"

    @pytest.mark.asyncio
    async def test_etf_variable_generation(self, analyzer):
        """
        RED TEST: ETF analysis should generate ETF-specific variables.
        This test will FAIL initially because ETF variable logic doesn't exist.
        """
        analysis = await analyzer.analyze_stock("IWM")
        
        # Mock report generator to access variable preparation
        generator = ReportGenerator()
        
        # This will fail initially because _prepare_etf_variables doesn't exist
        variables = generator._prepare_etf_variables(analysis)
        
        required_etf_vars = [
            "EXPENSE_RATIO",
            "TRACKING_ERROR", 
            "AUM",
            "HOLDINGS_COUNT",
            "DIVIDEND_YIELD"
        ]
        
        missing_vars = [var for var in required_etf_vars if var not in variables]
        assert len(missing_vars) == 0, f"Missing ETF-specific variables: {missing_vars}"


class TestAssetTypeDetection:
    """Test suite for asset type detection logic."""
    
    def test_detect_etf_symbols(self):
        """RED TEST: Should detect common ETF symbols."""
        analyzer = ComprehensiveStockAnalyzer()
        
        etf_symbols = ["IWM", "SPY", "QQQ", "VTI", "EFA", "GLD", "TLT"]
        
        for symbol in etf_symbols:
            # This will fail initially
            asset_type = analyzer.detect_asset_type(symbol)
            assert asset_type == "ETF", f"{symbol} should be detected as ETF"

    def test_detect_stock_symbols(self):
        """RED TEST: Should detect regular stock symbols.""" 
        analyzer = ComprehensiveStockAnalyzer()
        
        stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        for symbol in stock_symbols:
            # This will fail initially  
            asset_type = analyzer.detect_asset_type(symbol)
            assert asset_type == "STOCK", f"{symbol} should be detected as STOCK"


if __name__ == "__main__":
    # Run tests to see current failures (RED phase)
    pytest.main([__file__, "-v", "--tb=short"])