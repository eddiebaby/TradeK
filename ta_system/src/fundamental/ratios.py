"""Financial ratio calculations for fundamental analysis."""

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, Optional

from .models import (
    BalanceSheet,
    CashFlowStatement,
    FinancialRatios,
    IncomeStatement,
    MarketData,
)


class FinancialRatioCalculator:
    """Calculator for comprehensive financial ratios."""

    @staticmethod
    def _safe_divide(numerator: Optional[Decimal], denominator: Optional[Decimal]) -> Optional[Decimal]:
        """Safely divide two decimals, handling None and zero cases."""
        if numerator is None or denominator is None or denominator == 0:
            return None
        return (numerator / denominator).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    @staticmethod
    def _safe_multiply(a: Optional[Decimal], b: Optional[Decimal]) -> Optional[Decimal]:
        """Safely multiply two decimals, handling None cases."""
        if a is None or b is None:
            return None
        return a * b

    def calculate_profitability_ratios(
        self,
        income_statement: IncomeStatement,
        balance_sheet: BalanceSheet
    ) -> Dict[str, Optional[Decimal]]:
        """Calculate profitability ratios."""
        ratios = {}

        # Gross Margin = Gross Profit / Revenue
        ratios["gross_margin"] = self._safe_divide(income_statement.gross_profit, income_statement.revenue)
        if ratios["gross_margin"]:
            ratios["gross_margin"] *= 100  # Convert to percentage

        # Operating Margin = Operating Income / Revenue
        ratios["operating_margin"] = self._safe_divide(income_statement.operating_income, income_statement.revenue)
        if ratios["operating_margin"]:
            ratios["operating_margin"] *= 100

        # Net Margin = Net Income / Revenue
        ratios["net_margin"] = self._safe_divide(income_statement.net_income, income_statement.revenue)
        if ratios["net_margin"]:
            ratios["net_margin"] *= 100

        # Return on Assets (ROA) = Net Income / Total Assets
        ratios["return_on_assets"] = self._safe_divide(income_statement.net_income, balance_sheet.total_assets)
        if ratios["return_on_assets"]:
            ratios["return_on_assets"] *= 100

        # Return on Equity (ROE) = Net Income / Shareholders' Equity
        ratios["return_on_equity"] = self._safe_divide(income_statement.net_income, balance_sheet.shareholders_equity)
        if ratios["return_on_equity"]:
            ratios["return_on_equity"] *= 100

        # Return on Invested Capital (ROIC) = Operating Income / (Total Assets - Current Liabilities)
        if balance_sheet.current_liabilities:
            invested_capital = balance_sheet.total_assets - balance_sheet.current_liabilities
            ratios["return_on_invested_capital"] = self._safe_divide(income_statement.operating_income, invested_capital)
            if ratios["return_on_invested_capital"]:
                ratios["return_on_invested_capital"] *= 100
        else:
            ratios["return_on_invested_capital"] = None

        return ratios

    def calculate_liquidity_ratios(self, balance_sheet: BalanceSheet) -> Dict[str, Optional[Decimal]]:
        """Calculate liquidity ratios."""
        ratios = {}

        # Current Ratio = Current Assets / Current Liabilities
        ratios["current_ratio"] = self._safe_divide(balance_sheet.current_assets, balance_sheet.current_liabilities)

        # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
        if balance_sheet.current_assets and balance_sheet.inventory:
            quick_assets = balance_sheet.current_assets - balance_sheet.inventory
            ratios["quick_ratio"] = self._safe_divide(quick_assets, balance_sheet.current_liabilities)
        else:
            ratios["quick_ratio"] = None

        # Cash Ratio = Cash and Equivalents / Current Liabilities
        ratios["cash_ratio"] = self._safe_divide(balance_sheet.cash_and_equivalents, balance_sheet.current_liabilities)

        return ratios

    def calculate_leverage_ratios(
        self,
        income_statement: IncomeStatement,
        balance_sheet: BalanceSheet
    ) -> Dict[str, Optional[Decimal]]:
        """Calculate leverage ratios."""
        ratios = {}

        # Debt-to-Equity = Total Debt / Shareholders' Equity
        ratios["debt_to_equity"] = self._safe_divide(balance_sheet.total_debt, balance_sheet.shareholders_equity)

        # Debt-to-Assets = Total Debt / Total Assets
        ratios["debt_to_assets"] = self._safe_divide(balance_sheet.total_debt, balance_sheet.total_assets)

        # Interest Coverage Ratio = Operating Income / Interest Expense
        ratios["interest_coverage"] = self._safe_divide(income_statement.operating_income, income_statement.interest_expense)

        return ratios

    def calculate_efficiency_ratios(
        self,
        income_statement: IncomeStatement,
        balance_sheet: BalanceSheet
    ) -> Dict[str, Optional[Decimal]]:
        """Calculate efficiency ratios."""
        ratios = {}

        # Asset Turnover = Revenue / Total Assets
        ratios["asset_turnover"] = self._safe_divide(income_statement.revenue, balance_sheet.total_assets)

        # Inventory Turnover = Cost of Revenue / Inventory
        ratios["inventory_turnover"] = self._safe_divide(income_statement.cost_of_revenue, balance_sheet.inventory)

        # Receivables Turnover = Revenue / Accounts Receivable
        ratios["receivables_turnover"] = self._safe_divide(income_statement.revenue, balance_sheet.accounts_receivable)

        return ratios

    def calculate_valuation_ratios(
        self,
        income_statement: IncomeStatement,
        market_data: MarketData,
        growth_rate: Optional[Decimal] = None
    ) -> Dict[str, Optional[Decimal]]:
        """Calculate valuation ratios."""
        ratios = {}

        # Price-to-Earnings (P/E) = Market Price / EPS
        ratios["price_to_earnings"] = self._safe_divide(market_data.current_price, income_statement.eps_diluted)

        # Price-to-Book (P/B) = Market Price / Book Value per Share
        if market_data.shares_outstanding:
            book_value_per_share = self._safe_divide(
                market_data.market_cap,  # Using market cap as proxy
                market_data.shares_outstanding
            )
            ratios["price_to_book"] = self._safe_divide(market_data.current_price, book_value_per_share)
        else:
            ratios["price_to_book"] = None

        # Price-to-Sales (P/S) = Market Cap / Revenue
        ratios["price_to_sales"] = self._safe_divide(market_data.market_cap, income_statement.revenue)

        # Enterprise Value to Revenue = (Market Cap + Total Debt - Cash) / Revenue
        # Note: Simplified calculation without debt and cash
        ratios["enterprise_value_to_revenue"] = ratios["price_to_sales"]  # Approximation

        # Enterprise Value to EBITDA
        if income_statement.ebitda:
            ratios["enterprise_value_to_ebitda"] = self._safe_divide(market_data.market_cap, income_statement.ebitda)
        else:
            ratios["enterprise_value_to_ebitda"] = None

        # PEG Ratio = P/E Ratio / Growth Rate
        if ratios["price_to_earnings"] and growth_rate and growth_rate > 0:
            ratios["peg_ratio"] = self._safe_divide(ratios["price_to_earnings"], growth_rate)
        else:
            ratios["peg_ratio"] = None

        return ratios

    def calculate_all_ratios(
        self,
        symbol: str,
        income_statement: IncomeStatement,
        balance_sheet: BalanceSheet,
        market_data: MarketData,
        cash_flow_statement: Optional[CashFlowStatement] = None,
        growth_rate: Optional[Decimal] = None
    ) -> FinancialRatios:
        """Calculate comprehensive financial ratios."""
        
        # Calculate all ratio categories
        profitability = self.calculate_profitability_ratios(income_statement, balance_sheet)
        liquidity = self.calculate_liquidity_ratios(balance_sheet)
        leverage = self.calculate_leverage_ratios(income_statement, balance_sheet)
        efficiency = self.calculate_efficiency_ratios(income_statement, balance_sheet)
        valuation = self.calculate_valuation_ratios(income_statement, market_data, growth_rate)

        # Combine all ratios
        all_ratios = {**profitability, **liquidity, **leverage, **efficiency, **valuation}

        return FinancialRatios(
            symbol=symbol,
            as_of_date=datetime.now(),
            **all_ratios
        )

    def _create_default_ratios(self, symbol: str) -> FinancialRatios:
        """Create default financial ratios when data is unavailable."""
        return FinancialRatios(
            symbol=symbol,
            as_of_date=datetime.now(),
            gross_margin=Decimal("40.0"),
            operating_margin=Decimal("20.0"),
            net_margin=Decimal("15.0"),
            return_on_assets=Decimal("10.0"),
            return_on_equity=Decimal("15.0"),
            current_ratio=Decimal("2.0"),
            debt_to_equity=Decimal("0.5"),
            price_to_earnings=Decimal("20.0"),
            price_to_book=Decimal("3.0"),
            price_to_sales=Decimal("5.0"),
        )

    def calculate_growth_rates(
        self,
        current: IncomeStatement,
        previous: IncomeStatement
    ) -> Dict[str, Optional[Decimal]]:
        """Calculate year-over-year growth rates."""
        growth_rates = {}

        # Revenue Growth
        growth_rates["revenue_growth"] = self._calculate_growth_rate(current.revenue, previous.revenue)

        # Net Income Growth
        growth_rates["net_income_growth"] = self._calculate_growth_rate(current.net_income, previous.net_income)

        # Operating Income Growth
        growth_rates["operating_income_growth"] = self._calculate_growth_rate(
            current.operating_income, previous.operating_income
        )

        # EPS Growth
        growth_rates["eps_growth"] = self._calculate_growth_rate(current.eps_diluted, previous.eps_diluted)

        return growth_rates

    def _calculate_growth_rate(
        self,
        current: Optional[Decimal],
        previous: Optional[Decimal]
    ) -> Optional[Decimal]:
        """Calculate growth rate between two periods."""
        if current is None or previous is None or previous == 0:
            return None
        
        growth_rate = ((current - previous) / abs(previous)) * 100
        return growth_rate.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def benchmark_ratios(
        self,
        ratios: FinancialRatios,
        industry_benchmarks: Optional[Dict[str, Decimal]] = None
    ) -> Dict[str, str]:
        """Benchmark ratios against industry averages."""
        if not industry_benchmarks:
            return {}

        benchmarks = {}
        
        # Compare key ratios
        if ratios.return_on_equity and "roe" in industry_benchmarks:
            if ratios.return_on_equity > industry_benchmarks["roe"] * Decimal("1.2"):
                benchmarks["roe"] = "Above Average"
            elif ratios.return_on_equity < industry_benchmarks["roe"] * Decimal("0.8"):
                benchmarks["roe"] = "Below Average"
            else:
                benchmarks["roe"] = "Average"

        if ratios.price_to_earnings and "pe" in industry_benchmarks:
            if ratios.price_to_earnings > industry_benchmarks["pe"] * Decimal("1.3"):
                benchmarks["pe"] = "Expensive"
            elif ratios.price_to_earnings < industry_benchmarks["pe"] * Decimal("0.7"):
                benchmarks["pe"] = "Cheap"
            else:
                benchmarks["pe"] = "Fair Value"

        return benchmarks

    def get_ratio_interpretation(self, ratios: FinancialRatios) -> Dict[str, str]:
        """Provide interpretation of key financial ratios."""
        interpretations = {}

        # ROE Interpretation
        if ratios.return_on_equity:
            if ratios.return_on_equity > 20:
                interpretations["roe"] = "Excellent"
            elif ratios.return_on_equity > 15:
                interpretations["roe"] = "Good"
            elif ratios.return_on_equity > 10:
                interpretations["roe"] = "Average"
            else:
                interpretations["roe"] = "Poor"

        # Current Ratio Interpretation
        if ratios.current_ratio:
            if ratios.current_ratio > 2:
                interpretations["current_ratio"] = "Strong Liquidity"
            elif ratios.current_ratio > 1:
                interpretations["current_ratio"] = "Adequate Liquidity"
            else:
                interpretations["current_ratio"] = "Liquidity Concern"

        # Debt-to-Equity Interpretation
        if ratios.debt_to_equity:
            if ratios.debt_to_equity > 2:
                interpretations["debt_to_equity"] = "High Leverage"
            elif ratios.debt_to_equity > 1:
                interpretations["debt_to_equity"] = "Moderate Leverage"
            else:
                interpretations["debt_to_equity"] = "Conservative Leverage"

        return interpretations