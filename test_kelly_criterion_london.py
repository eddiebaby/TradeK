#!/usr/bin/env python3
"""
London School TDD Tests for Kelly Criterion Position Sizing
===========================================================

Outside-in behavior-driven tests focusing on user stories
and component collaborations with mocks for Kelly Criterion implementation.

Kelly Criterion Formula: f* = (bp - q) / b
Where:
- f* = optimal fraction of capital to wager
- b = odds received on the wager (net odds)  
- p = probability of winning
- q = probability of losing (1 - p)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import json
import numpy as np
from typing import Dict, Any

# Import will fail initially - this is expected in London TDD RED phase
try:
    from kelly_criterion_calculator import KellyCriterionCalculator, ProbabilityEstimator, PortfolioManager
except ImportError:
    # Mock the classes for RED phase
    class KellyCriterionCalculator:
        pass
    class ProbabilityEstimator:
        pass  
    class PortfolioManager:
        pass

class UserStory:
    """Helper class for expressing user stories"""
    def __init__(self, title: str, as_a: str, i_want: str, so_that: str):
        self.title = title
        self.as_a = as_a
        self.i_want = i_want
        self.so_that = so_that
        
    def __str__(self):
        return f"{self.title}\nAs a {self.as_a}\nI want {self.i_want}\nSo that {self.so_that}"

class TestKellyCriterionUserStories:
    """London School TDD - Start with user behavior"""
    
    def test_trader_can_calculate_optimal_position_size(self):
        """
        GIVEN a trader knows win probability and odds
        WHEN they calculate Kelly position size
        THEN they receive optimal capital fraction
        """
        story = UserStory(
            title="Calculate Optimal Position Size",
            as_a="quantitative trader",
            i_want="to calculate optimal position sizes using Kelly Criterion",
            so_that="I maximize long-term growth while managing risk"
        )
        
        # Arrange - Mock the dependencies
        mock_prob_estimator = Mock()
        mock_prob_estimator.estimate_win_probability.return_value = 0.6  # 60% win rate
        
        calculator = KellyCriterionCalculator(probability_estimator=mock_prob_estimator)
        
        # Act - Calculate Kelly position
        optimal_fraction = calculator.calculate_kelly_fraction(
            odds=2.0,  # 2:1 odds (win $2 for every $1 bet)
            win_probability=0.6
        )
        
        # Assert - Verify Kelly formula: f* = (bp - q) / b = (2*0.6 - 0.4) / 2 = 0.4
        expected_fraction = (2.0 * 0.6 - 0.4) / 2.0
        assert optimal_fraction == pytest.approx(expected_fraction, rel=1e-9)
        assert optimal_fraction == pytest.approx(0.4, rel=1e-9)  # 40% of capital
    
    def test_trader_gets_zero_for_unfavorable_odds(self):
        """
        GIVEN a trader faces unfavorable odds (negative expected value)
        WHEN they calculate Kelly position size
        THEN they receive zero position (don't bet)
        """
        story = UserStory(
            title="Handle Unfavorable Odds",
            as_a="risk-conscious trader", 
            i_want="to get zero position size for negative expected value bets",
            so_that="I don't lose money on bad bets"
        )
        
        calculator = KellyCriterionCalculator()
        
        # Unfavorable odds: 40% win rate with 1:1 odds
        optimal_fraction = calculator.calculate_kelly_fraction(
            odds=1.0,  # 1:1 odds  
            win_probability=0.4  # 40% win rate
        )
        
        # Kelly = (1*0.4 - 0.6) / 1 = -0.2, should be clamped to 0
        assert optimal_fraction == 0.0
    
    def test_portfolio_manager_applies_kelly_to_portfolio(self):
        """
        GIVEN a portfolio manager has multiple trading opportunities
        WHEN they apply Kelly sizing to the portfolio
        THEN each position is optimally sized based on Kelly calculation
        """
        story = UserStory(
            title="Portfolio Kelly Sizing",
            as_a="portfolio manager",
            i_want="to apply Kelly sizing across multiple positions", 
            so_that="I optimize capital allocation for the entire portfolio"
        )
        
        # Mock dependencies
        mock_portfolio_manager = Mock()
        mock_portfolio_manager.get_available_capital.return_value = 100000  # $100k
        
        calculator = KellyCriterionCalculator()
        
        # Multiple opportunities
        opportunities = [
            {"symbol": "AAPL", "odds": 1.5, "win_prob": 0.65},
            {"symbol": "GOOGL", "odds": 2.0, "win_prob": 0.55},
            {"symbol": "MSFT", "odds": 1.2, "win_prob": 0.7}
        ]
        
        position_sizes = calculator.calculate_portfolio_positions(
            opportunities=opportunities,
            total_capital=100000,
            portfolio_manager=mock_portfolio_manager
        )
        
        # Verify we get position sizes for each symbol
        assert "AAPL" in position_sizes
        assert "GOOGL" in position_sizes
        assert "MSFT" in position_sizes
        
        # Verify position sizes are reasonable (between 0 and total capital)
        for symbol, size in position_sizes.items():
            assert 0 <= size <= 100000
            assert isinstance(size, (int, float))

class TestKellyCriterionCollaborations:
    """Test interactions between components using mocks"""
    
    def test_kelly_calculator_delegates_to_probability_estimator(self):
        """Verify correct delegation to ProbabilityEstimator"""
        # Mock ProbabilityEstimator
        mock_prob_estimator = Mock()
        mock_prob_estimator.estimate_win_probability.return_value = 0.55
        
        calculator = KellyCriterionCalculator(probability_estimator=mock_prob_estimator)
        
        # Request probability estimation
        historical_data = [1, -1, 1, 1, -1]  # Mock price movements
        calculator.calculate_kelly_with_estimation(
            odds=1.8,
            historical_data=historical_data
        )
        
        # Verify delegation
        mock_prob_estimator.estimate_win_probability.assert_called_once_with(historical_data)
    
    def test_kelly_calculator_integrates_with_portfolio_manager(self):
        """Verify integration with PortfolioManager"""
        # Mock PortfolioManager
        mock_portfolio = Mock()
        mock_portfolio.get_current_positions.return_value = {"AAPL": 1000, "GOOGL": 500}
        mock_portfolio.get_available_capital.return_value = 50000
        
        calculator = KellyCriterionCalculator()
        
        # Apply Kelly sizing to existing portfolio
        new_positions = calculator.rebalance_portfolio(
            target_allocations={"AAPL": 0.3, "GOOGL": 0.2, "TSLA": 0.1},
            portfolio_manager=mock_portfolio
        )
        
        # Verify portfolio manager was consulted
        mock_portfolio.get_current_positions.assert_called_once()
        mock_portfolio.get_available_capital.assert_called_once()
        
        # Verify new positions calculated
        assert isinstance(new_positions, dict)

class TestKellyCriterionMathematicalValidation:
    """Test mathematical correctness of Kelly Criterion implementation"""
    
    def test_kelly_formula_mathematical_correctness(self):
        """Verify Kelly formula implementation matches mathematical definition"""
        calculator = KellyCriterionCalculator()
        
        # Test case 1: Standard favorable bet
        # f* = (bp - q) / b = (1.5*0.6 - 0.4) / 1.5 = (0.9 - 0.4) / 1.5 = 0.5 / 1.5 = 0.333
        result = calculator.calculate_kelly_fraction(odds=1.5, win_probability=0.6)
        expected = (1.5 * 0.6 - 0.4) / 1.5
        assert result == pytest.approx(expected, rel=1e-9)
        
        # Test case 2: Edge case - 50% probability
        result = calculator.calculate_kelly_fraction(odds=2.0, win_probability=0.5)
        expected = (2.0 * 0.5 - 0.5) / 2.0  # = 0.25
        assert result == pytest.approx(expected, rel=1e-9)
        
        # Test case 3: High confidence bet
        result = calculator.calculate_kelly_fraction(odds=1.2, win_probability=0.9)
        expected = (1.2 * 0.9 - 0.1) / 1.2  # = 0.9833...
        assert result == pytest.approx(expected, rel=1e-9)

class TestKellyCriterionEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_handles_extreme_probabilities(self):
        """Test boundary conditions for probabilities"""
        calculator = KellyCriterionCalculator()
        
        # Probability = 0 (certain loss)
        result = calculator.calculate_kelly_fraction(odds=10.0, win_probability=0.0)
        assert result == 0.0
        
        # Probability = 1 (certain win) - should be maximum allocation
        result = calculator.calculate_kelly_fraction(odds=1.1, win_probability=1.0)
        expected = (1.1 * 1.0 - 0.0) / 1.1  # = 1.0 (100% allocation)
        assert result == pytest.approx(expected, rel=1e-9)
    
    def test_handles_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        calculator = KellyCriterionCalculator()
        
        with pytest.raises(ValueError):
            calculator.calculate_kelly_fraction(odds=-1.0, win_probability=0.5)  # Negative odds
            
        with pytest.raises(ValueError):
            calculator.calculate_kelly_fraction(odds=1.0, win_probability=1.5)  # Probability > 1
            
        with pytest.raises(ValueError):
            calculator.calculate_kelly_fraction(odds=1.0, win_probability=-0.1)  # Negative probability

class TestKellyCriterionKnowledgeBaseIntegration:
    """Test integration with local knowledge base"""
    
    def test_finds_kelly_criterion_in_knowledge_base(self):
        """Verify system can find Kelly Criterion concept in local books"""
        # This test validates our local AI system knowledge base
        from local_ai_trading_system import LocalBookSearch
        
        book_search = LocalBookSearch()
        
        # Search for Kelly Criterion concept
        context = book_search.search_relevant_context("Kelly criterion position sizing")
        
        # Verify Kelly Criterion is found in risk management concepts
        assert "Kelly criterion calculation" in context or "kelly" in context.lower()
        assert "risk_management" in book_search.knowledge_base["concepts"]
        
        # Verify risk management concepts include position sizing
        risk_concepts = book_search.knowledge_base["concepts"]["risk_management"]["concepts"]
        kelly_found = any("Kelly" in concept for concept in risk_concepts)
        position_sizing_found = any("position sizing" in concept.lower() for concept in risk_concepts)
        
        assert kelly_found or position_sizing_found, "Kelly Criterion or position sizing should be in risk management concepts"

class TestKellyCriterionPerformanceMetrics:
    """Test performance characteristics"""
    
    def test_calculation_speed(self):
        """Verify Kelly calculations are fast enough for trading use"""
        import time
        
        calculator = KellyCriterionCalculator()
        
        start_time = time.time()
        
        # Perform 1000 calculations
        for i in range(1000):
            calculator.calculate_kelly_fraction(
                odds=1.5 + (i % 10) * 0.1,
                win_probability=0.5 + (i % 5) * 0.1
            )
        
        end_time = time.time()
        calculation_time = end_time - start_time
        
        # Should complete 1000 calculations in under 0.1 seconds
        assert calculation_time < 0.1, f"Kelly calculations too slow: {calculation_time}s for 1000 calculations"

if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short"])