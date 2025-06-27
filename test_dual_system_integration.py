#!/usr/bin/env python3
"""
Integration Tests for Kelly Criterion + HFT Business Analysis
=============================================================

London School TDD integration tests validating both systems
work together and with the local AI knowledge base.
"""

import pytest
from unittest.mock import Mock, patch
from kelly_criterion_calculator import KellyCriterionCalculator
from hft_business_analyzer import HFTBusinessAnalyzer
from local_ai_trading_system import LocalBookSearch

class TestKellyHFTIntegration:
    """Test integration between Kelly Criterion and HFT Business systems"""
    
    def test_kelly_sizing_for_hft_strategies(self):
        """Test Kelly position sizing applied to HFT trading strategies"""
        # Initialize both systems
        kelly_calc = KellyCriterionCalculator()
        hft_analyzer = HFTBusinessAnalyzer()
        
        # Get HFT revenue models
        revenue_models = hft_analyzer.analyze_revenue_models()
        
        # Create trading opportunities based on HFT strategies
        hft_opportunities = []
        for strategy_name, strategy_data in revenue_models.items():
            # Mock probability and odds for each HFT strategy
            if "market_making" in strategy_name:
                opportunity = {"symbol": "MM_STRATEGY", "odds": 1.1, "win_prob": 0.65}
            elif "arbitrage" in strategy_name:
                opportunity = {"symbol": "ARB_STRATEGY", "odds": 1.2, "win_prob": 0.70}
            else:
                opportunity = {"symbol": "STAT_STRATEGY", "odds": 1.5, "win_prob": 0.55}
            
            hft_opportunities.append(opportunity)
        
        # Apply Kelly sizing to HFT portfolio
        total_capital = 1000000  # $1M capital
        kelly_positions = kelly_calc.calculate_portfolio_positions(
            hft_opportunities, total_capital
        )
        
        # Verify integration
        assert len(kelly_positions) >= 3  # Should have positions for multiple strategies
        
        total_allocation = sum(kelly_positions.values())
        assert total_allocation <= total_capital  # Don't over-allocate
        
        # Verify each position is reasonable for HFT
        for strategy, position_size in kelly_positions.items():
            assert position_size >= 0  # No negative positions
            assert position_size <= total_capital * 0.5  # No single position > 50%
    
    def test_hft_business_intelligence_informs_kelly_parameters(self):
        """Test HFT business analysis informing Kelly calculation parameters"""
        hft_analyzer = HFTBusinessAnalyzer()
        kelly_calc = KellyCriterionCalculator()
        
        # Get HFT competitive analysis
        competitive_analysis = hft_analyzer.analyze_competitive_landscape()
        
        # Business intelligence should inform risk parameters
        assert "technology_advantages" in competitive_analysis
        assert "capital_requirements" in competitive_analysis
        
        # Use business insights to adjust Kelly calculations
        # High-tech advantage suggests higher win probability
        tech_advantage_factor = 1.1  # 10% boost for tech advantage
        base_win_prob = 0.55
        adjusted_win_prob = min(0.95, base_win_prob * tech_advantage_factor)
        
        # Calculate Kelly with business-informed parameters
        kelly_fraction = kelly_calc.calculate_kelly_fraction(
            odds=1.3, 
            win_probability=adjusted_win_prob
        )
        
        # Verify reasonable result
        assert 0 <= kelly_fraction <= 1
        assert kelly_fraction > 0  # Should be profitable with tech advantage

class TestLocalKnowledgeBaseIntegration:
    """Test both systems with local knowledge base"""
    
    def test_knowledge_base_supports_both_domains(self):
        """Verify knowledge base contains concepts for both Kelly and HFT"""
        book_search = LocalBookSearch()
        
        # Test Kelly Criterion knowledge
        kelly_context = book_search.search_relevant_context("Kelly criterion position sizing")
        assert len(kelly_context) > 50  # Substantial Kelly content
        
        # Test HFT business knowledge  
        hft_context = book_search.search_relevant_context("high-frequency trading business")
        assert len(hft_context) > 50  # Substantial HFT content
        
        # Verify knowledge base structure
        assert "risk_management" in book_search.knowledge_base["concepts"]
        assert "high_frequency_trading" in book_search.knowledge_base["concepts"]
        
        # Verify concept quality
        risk_concepts = book_search.knowledge_base["concepts"]["risk_management"]["concepts"]
        hft_concepts = book_search.knowledge_base["concepts"]["high_frequency_trading"]["concepts"]
        
        assert len(risk_concepts) >= 3
        assert len(hft_concepts) >= 3
    
    def test_offline_ai_handles_both_concept_types(self):
        """Test offline AI system handles both mathematical and business queries"""
        from claude_code_offline_mode import ClaudeCodeOfflineMode
        
        offline_mode = ClaudeCodeOfflineMode()
        
        # Test mathematical query (Kelly Criterion)
        kelly_result = offline_mode.handle_request("calculate Kelly criterion position size")
        assert kelly_result["success"] == True
        assert "kelly" in kelly_result["content"].lower() or "position" in kelly_result["content"].lower()
        
        # Test business query (HFT)
        hft_result = offline_mode.handle_request("high-frequency trading business overview")
        assert hft_result["success"] == True
        assert "trading" in hft_result["content"].lower() or "frequency" in hft_result["content"].lower()
        
        # Verify both use zero tokens
        assert kelly_result.get("cost", 0) == 0
        assert hft_result.get("cost", 0) == 0

class TestProductionWorkflow:
    """Test realistic production workflow using both systems"""
    
    def test_complete_hft_strategy_development_workflow(self):
        """Test end-to-end workflow: business analysis → Kelly sizing → implementation"""
        # Step 1: Business Analysis
        hft_analyzer = HFTBusinessAnalyzer()
        business_overview = hft_analyzer.generate_business_overview("HFT market making")
        
        assert "market_microstructure" in business_overview
        assert "revenue_models" in business_overview
        
        # Step 2: Strategy Selection (based on business analysis)
        revenue_models = hft_analyzer.analyze_revenue_models()
        selected_strategy = "market_making"  # Choose based on analysis
        
        assert selected_strategy in revenue_models
        strategy_data = revenue_models[selected_strategy]
        assert "profit_mechanism" in strategy_data
        
        # Step 3: Kelly Position Sizing
        kelly_calc = KellyCriterionCalculator()
        
        # Market making typically has high win rate, low odds
        kelly_fraction = kelly_calc.calculate_kelly_fraction(
            odds=1.05,  # Small spread capture
            win_probability=0.75  # High success rate for market making
        )
        
        # Step 4: Risk Management
        max_position_limit = 0.25  # 25% maximum position
        final_position_size = min(kelly_fraction, max_position_limit)
        
        # Verify realistic results
        assert 0 <= final_position_size <= max_position_limit
        assert final_position_size > 0  # Should be profitable
        
        # Step 5: Performance Metrics
        expected_return = kelly_fraction * 0.05 * 0.75  # Expected profit
        assert expected_return > 0
    
    def test_multi_strategy_hft_portfolio(self):
        """Test portfolio of multiple HFT strategies with Kelly sizing"""
        hft_analyzer = HFTBusinessAnalyzer()
        kelly_calc = KellyCriterionCalculator()
        
        # Get all HFT revenue models
        revenue_models = hft_analyzer.analyze_revenue_models()
        
        # Create portfolio opportunities
        portfolio_opportunities = []
        strategy_mapping = {
            "market_making": {"odds": 1.05, "win_prob": 0.75},
            "statistical_arbitrage": {"odds": 1.3, "win_prob": 0.60}, 
            "cross_venue_arbitrage": {"odds": 1.15, "win_prob": 0.65},
            "news_trading": {"odds": 2.0, "win_prob": 0.45}
        }
        
        for strategy_name in revenue_models.keys():
            if strategy_name in strategy_mapping:
                params = strategy_mapping[strategy_name]
                opportunity = {
                    "symbol": strategy_name.upper(),
                    "odds": params["odds"],
                    "win_prob": params["win_prob"]
                }
                portfolio_opportunities.append(opportunity)
        
        # Calculate Kelly portfolio
        total_capital = 10000000  # $10M HFT fund
        kelly_portfolio = kelly_calc.calculate_portfolio_positions(
            portfolio_opportunities, total_capital
        )
        
        # Verify portfolio characteristics
        assert len(kelly_portfolio) >= 3  # Diversified portfolio
        
        total_allocation = sum(kelly_portfolio.values())
        assert total_allocation <= total_capital
        
        # Market making should get significant allocation (high win rate)
        if "MARKET_MAKING" in kelly_portfolio:
            mm_allocation = kelly_portfolio["MARKET_MAKING"]
            assert mm_allocation > 0
            
        # Verify allocation percentages are reasonable
        for strategy, allocation in kelly_portfolio.items():
            allocation_pct = allocation / total_capital
            assert 0 <= allocation_pct <= 0.5  # No strategy > 50%

if __name__ == "__main__":
    pytest.main([__file__, "-v"])