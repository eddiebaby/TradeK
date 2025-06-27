#!/usr/bin/env python3
"""
Kelly Criterion Position Sizing Calculator
==========================================

Implements the Kelly Criterion for optimal position sizing in trading.

Kelly Criterion Formula: f* = (bp - q) / b
Where:
- f* = optimal fraction of capital to wager
- b = odds received on the wager (net odds)
- p = probability of winning  
- q = probability of losing (1 - p)

This implementation follows London School TDD - minimal code to pass tests.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ProbabilityEstimator:
    """Estimates win probabilities from historical data"""
    
    def estimate_win_probability(self, historical_data: List[float]) -> float:
        """Estimate win probability from historical price movements"""
        if not historical_data:
            return 0.5  # Default 50% probability
        
        wins = sum(1 for x in historical_data if x > 0)
        total = len(historical_data)
        
        return wins / total if total > 0 else 0.5

class PortfolioManager:
    """Manages portfolio positions and capital allocation"""
    
    def __init__(self):
        self.current_positions = {}
        self.available_capital = 100000  # Default $100k
    
    def get_current_positions(self) -> Dict[str, float]:
        """Get current portfolio positions"""
        return self.current_positions.copy()
    
    def get_available_capital(self) -> float:
        """Get available capital for new positions"""
        return self.available_capital

class KellyCriterionCalculator:
    """Kelly Criterion calculator for optimal position sizing"""
    
    def __init__(self, probability_estimator: Optional[ProbabilityEstimator] = None):
        self.probability_estimator = probability_estimator or ProbabilityEstimator()
        logger.info("Kelly Criterion Calculator initialized")
    
    def calculate_kelly_fraction(self, odds: float, win_probability: float) -> float:
        """
        Calculate optimal Kelly fraction
        
        Args:
            odds: Net odds (e.g., 1.5 means win $1.50 for every $1 bet)
            win_probability: Probability of winning (0 to 1)
            
        Returns:
            Optimal fraction of capital to allocate (0 to 1)
        """
        # Input validation
        if odds <= 0:
            raise ValueError("Odds must be positive")
        if not 0 <= win_probability <= 1:
            raise ValueError("Win probability must be between 0 and 1")
        
        # Kelly formula: f* = (bp - q) / b
        # where q = 1 - p (probability of losing)
        lose_probability = 1 - win_probability
        kelly_fraction = (odds * win_probability - lose_probability) / odds
        
        # Never recommend negative positions (don't bet on negative expected value)
        return max(0.0, kelly_fraction)
    
    def calculate_kelly_with_estimation(self, odds: float, historical_data: List[float]) -> float:
        """Calculate Kelly fraction using estimated probability from historical data"""
        estimated_probability = self.probability_estimator.estimate_win_probability(historical_data)
        return self.calculate_kelly_fraction(odds, estimated_probability)
    
    def calculate_portfolio_positions(self, opportunities: List[Dict[str, Any]], 
                                    total_capital: float,
                                    portfolio_manager: Optional[PortfolioManager] = None) -> Dict[str, float]:
        """
        Calculate Kelly-optimal position sizes for portfolio of opportunities
        
        Args:
            opportunities: List of trading opportunities with 'symbol', 'odds', 'win_prob'
            total_capital: Total available capital
            portfolio_manager: Optional portfolio manager for advanced features
            
        Returns:
            Dictionary mapping symbols to position sizes in dollars
        """
        position_sizes = {}
        
        for opportunity in opportunities:
            symbol = opportunity["symbol"]
            odds = opportunity["odds"]
            win_prob = opportunity["win_prob"]
            
            # Calculate Kelly fraction for this opportunity
            kelly_fraction = self.calculate_kelly_fraction(odds, win_prob)
            
            # Convert to position size in dollars
            position_size = kelly_fraction * total_capital
            position_sizes[symbol] = position_size
        
        return position_sizes
    
    def rebalance_portfolio(self, target_allocations: Dict[str, float],
                          portfolio_manager: PortfolioManager) -> Dict[str, float]:
        """
        Rebalance portfolio based on target Kelly allocations
        
        Args:
            target_allocations: Target allocation fractions by symbol
            portfolio_manager: Portfolio manager providing current state
            
        Returns:
            New position sizes by symbol
        """
        available_capital = portfolio_manager.get_available_capital()
        current_positions = portfolio_manager.get_current_positions()
        
        new_positions = {}
        
        for symbol, target_fraction in target_allocations.items():
            # Calculate new position size based on available capital
            new_position_size = target_fraction * available_capital
            new_positions[symbol] = new_position_size
        
        return new_positions

def demo_kelly_calculator():
    """Demonstration of Kelly Criterion calculator"""
    print("🎯 Kelly Criterion Calculator Demo")
    print("=" * 50)
    
    calculator = KellyCriterionCalculator()
    
    # Example 1: Basic Kelly calculation
    print("\n📊 Example 1: Basic Kelly Calculation")
    odds = 1.5  # 1.5:1 odds
    win_prob = 0.6  # 60% win probability
    
    kelly_fraction = calculator.calculate_kelly_fraction(odds, win_prob)
    print(f"Odds: {odds}:1")
    print(f"Win Probability: {win_prob:.1%}")
    print(f"Kelly Fraction: {kelly_fraction:.3f} ({kelly_fraction:.1%} of capital)")
    
    # Example 2: Portfolio allocation
    print("\n📈 Example 2: Portfolio Allocation")
    opportunities = [
        {"symbol": "AAPL", "odds": 1.5, "win_prob": 0.65},
        {"symbol": "GOOGL", "odds": 2.0, "win_prob": 0.55},
        {"symbol": "MSFT", "odds": 1.2, "win_prob": 0.7}
    ]
    
    total_capital = 100000
    positions = calculator.calculate_portfolio_positions(opportunities, total_capital)
    
    print(f"Total Capital: ${total_capital:,}")
    for symbol, position_size in positions.items():
        fraction = position_size / total_capital
        print(f"{symbol}: ${position_size:,.0f} ({fraction:.1%})")
    
    # Example 3: Unfavorable odds (should return 0)
    print("\n⚠️  Example 3: Unfavorable Odds")
    unfavorable_kelly = calculator.calculate_kelly_fraction(odds=1.0, win_probability=0.4)
    print(f"Odds: 1:1, Win Prob: 40% → Kelly: {unfavorable_kelly:.3f} (Don't bet!)")
    
    print(f"\n✅ Kelly Calculator demo completed!")

if __name__ == "__main__":
    demo_kelly_calculator()