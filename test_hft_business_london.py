#!/usr/bin/env python3
"""
London School TDD Tests for HFT Business Overview System
========================================================

Outside-in behavior-driven tests focusing on user stories
and component collaborations for extracting High-Frequency Trading
business concepts from local knowledge base.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import json
from typing import Dict, Any, List

# Import will fail initially - this is expected in London TDD RED phase
try:
    from hft_business_analyzer import HFTBusinessAnalyzer, ConceptExtractor, BusinessIntelligenceReporter
except ImportError:
    # Mock the classes for RED phase
    class HFTBusinessAnalyzer:
        pass
    class ConceptExtractor:
        pass
    class BusinessIntelligenceReporter:
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

class TestHFTBusinessUserStories:
    """London School TDD - Start with user behavior"""
    
    def test_business_analyst_can_get_hft_overview(self):
        """
        GIVEN a business analyst wants to understand HFT
        WHEN they request an HFT business overview
        THEN they receive comprehensive business intelligence
        """
        story = UserStory(
            title="HFT Business Overview",
            as_a="business analyst",
            i_want="to get a comprehensive overview of high-frequency trading business",
            so_that="I understand the market structure and revenue models"
        )
        
        # Arrange
        analyzer = HFTBusinessAnalyzer()
        
        # Act
        business_overview = analyzer.generate_business_overview("high-frequency trading")
        
        # Assert - Should contain key HFT business areas
        assert "market_microstructure" in business_overview
        assert "revenue_models" in business_overview
        assert "technology_infrastructure" in business_overview
        assert "competitive_advantages" in business_overview
        
        # Verify content quality
        assert len(business_overview["market_microstructure"]) > 0
        assert "latency" in str(business_overview).lower()
        assert "liquidity" in str(business_overview).lower()
    
    def test_strategist_can_analyze_hft_revenue_models(self):
        """
        GIVEN a business strategist needs to understand HFT revenue
        WHEN they analyze HFT revenue models
        THEN they receive detailed revenue stream analysis
        """
        story = UserStory(
            title="HFT Revenue Analysis",
            as_a="business strategist",
            i_want="to analyze HFT revenue models and profit mechanisms",
            so_that="I can evaluate HFT business opportunities"
        )
        
        analyzer = HFTBusinessAnalyzer()
        
        # Request specific revenue analysis
        revenue_analysis = analyzer.analyze_revenue_models()
        
        # Verify revenue model categories
        expected_models = [
            "market_making", 
            "statistical_arbitrage",
            "cross_venue_arbitrage", 
            "news_trading"
        ]
        
        for model in expected_models:
            assert model in revenue_analysis, f"Missing revenue model: {model}"
        
        # Verify each model has description and profit mechanism
        for model_name, model_data in revenue_analysis.items():
            assert "description" in model_data
            assert "profit_mechanism" in model_data
            assert len(model_data["description"]) > 50  # Substantial description
    
    def test_executive_can_get_competitive_landscape(self):
        """
        GIVEN an executive needs competitive intelligence
        WHEN they request HFT competitive landscape analysis
        THEN they receive strategic market positioning insights
        """
        story = UserStory(
            title="HFT Competitive Analysis",
            as_a="executive",
            i_want="to understand the competitive landscape in HFT",
            so_that="I can make strategic business decisions"
        )
        
        analyzer = HFTBusinessAnalyzer()
        
        competitive_analysis = analyzer.analyze_competitive_landscape()
        
        # Verify competitive factors covered
        assert "technology_advantages" in competitive_analysis
        assert "capital_requirements" in competitive_analysis 
        assert "regulatory_barriers" in competitive_analysis
        assert "market_share_factors" in competitive_analysis
        
        # Verify analysis depth
        for factor, analysis in competitive_analysis.items():
            assert isinstance(analysis, (str, dict, list))
            assert len(str(analysis)) > 30  # Meaningful content

class TestHFTBusinessCollaborations:
    """Test interactions between components using mocks"""
    
    def test_hft_analyzer_delegates_to_concept_extractor(self):
        """Verify correct delegation to ConceptExtractor"""
        # Mock ConceptExtractor
        mock_extractor = Mock()
        mock_extractor.extract_concepts.return_value = {
            "market_making": ["bid-ask spread capture", "liquidity provision"],
            "latency_optimization": ["co-location", "FPGA acceleration"]
        }
        
        analyzer = HFTBusinessAnalyzer(concept_extractor=mock_extractor)
        
        # Request concept extraction
        concepts = analyzer.extract_business_concepts("HFT market making strategies")
        
        # Verify delegation occurred
        mock_extractor.extract_concepts.assert_called_once_with("HFT market making strategies")
        
        # Verify concepts returned
        assert "market_making" in concepts
        assert "latency_optimization" in concepts
    
    def test_hft_analyzer_integrates_with_business_reporter(self):
        """Verify integration with BusinessIntelligenceReporter"""
        # Mock BusinessIntelligenceReporter
        mock_reporter = Mock()
        mock_reporter.generate_report.return_value = {
            "executive_summary": "HFT is a technology-driven business...",
            "key_findings": ["Speed is crucial", "Capital efficiency matters"],
            "recommendations": ["Invest in technology", "Focus on risk management"]
        }
        
        analyzer = HFTBusinessAnalyzer(business_reporter=mock_reporter)
        
        # Generate business intelligence report
        report = analyzer.generate_intelligence_report("HFT business analysis")
        
        # Verify reporter was used
        mock_reporter.generate_report.assert_called_once()
        
        # Verify report structure
        assert "executive_summary" in report
        assert "key_findings" in report
        assert "recommendations" in report

class TestHFTBusinessKnowledgeExtraction:
    """Test knowledge extraction from local book database"""
    
    def test_extracts_hft_concepts_from_knowledge_base(self):
        """Verify system extracts HFT concepts from local knowledge"""
        from local_ai_trading_system import LocalBookSearch
        
        book_search = LocalBookSearch()
        analyzer = HFTBusinessAnalyzer(book_search=book_search)
        
        # Extract HFT business concepts
        concepts = analyzer.extract_hft_concepts()
        
        # Verify HFT knowledge base exists
        assert "high_frequency_trading" in book_search.knowledge_base["concepts"]
        
        # Verify key HFT concepts are extracted
        hft_concepts = book_search.knowledge_base["concepts"]["high_frequency_trading"]["concepts"]
        
        # Check for expected business concepts
        expected_concepts = [
            "market microstructure",
            "latency optimization", 
            "order book dynamics",
            "liquidity provision"
        ]
        
        concepts_text = " ".join(hft_concepts).lower()
        found_concepts = []
        for concept in expected_concepts:
            if any(word in concepts_text for word in concept.split()):
                found_concepts.append(concept)
        
        assert len(found_concepts) >= 2, f"Should find at least 2 HFT concepts, found: {found_concepts}"
    
    def test_searches_hft_business_context_correctly(self):
        """Test searching for HFT business context in knowledge base"""
        from local_ai_trading_system import LocalBookSearch
        
        book_search = LocalBookSearch()
        
        # Search for HFT business overview
        context = book_search.search_relevant_context("overview of high-frequency trading business")
        
        # Verify meaningful context returned
        assert len(context) > 100, "Should return substantial context for HFT business"
        
        # Verify business-relevant terms found
        business_terms = ["revenue", "profit", "business", "market", "trading", "technology"]
        context_lower = context.lower()
        
        found_terms = [term for term in business_terms if term in context_lower]
        assert len(found_terms) >= 3, f"Should find business terms in context, found: {found_terms}"

class TestHFTBusinessIntelligenceReporting:
    """Test business intelligence report generation"""
    
    def test_generates_comprehensive_business_report(self):
        """Test generation of comprehensive HFT business report"""
        analyzer = HFTBusinessAnalyzer()
        
        # Generate full business intelligence report
        report = analyzer.generate_comprehensive_report()
        
        # Verify report sections
        required_sections = [
            "executive_summary",
            "market_overview", 
            "revenue_models",
            "technology_infrastructure",
            "competitive_landscape",
            "risk_factors",
            "opportunities"
        ]
        
        for section in required_sections:
            assert section in report, f"Missing report section: {section}"
            assert len(str(report[section])) > 50, f"Section {section} too brief"
    
    def test_formats_business_metrics_correctly(self):
        """Test proper formatting of business metrics"""
        analyzer = HFTBusinessAnalyzer()
        
        # Get business metrics
        metrics = analyzer.get_business_metrics()
        
        # Verify metric categories
        expected_metrics = [
            "market_impact",
            "technology_costs",
            "revenue_streams", 
            "competitive_moats"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            
        # Verify metrics have proper structure
        for metric_name, metric_data in metrics.items():
            assert isinstance(metric_data, (dict, list, str))
            if isinstance(metric_data, dict):
                assert len(metric_data) > 0

class TestHFTBusinessEdgeCases:
    """Test edge cases and error handling"""
    
    def test_handles_unknown_business_queries(self):
        """Test handling of business queries not in knowledge base"""
        analyzer = HFTBusinessAnalyzer()
        
        # Query for non-existent concept
        result = analyzer.analyze_business_concept("quantum trading algorithms")
        
        # Should handle gracefully
        assert result is not None
        assert "unknown" in str(result).lower() or "not found" in str(result).lower()
    
    def test_handles_empty_knowledge_base(self):
        """Test behavior when knowledge base is empty or unavailable"""
        # Mock empty knowledge base
        mock_book_search = Mock()
        mock_book_search.knowledge_base = {"concepts": {}}
        mock_book_search.search_relevant_context.return_value = ""
        
        analyzer = HFTBusinessAnalyzer(book_search=mock_book_search)
        
        # Should handle empty knowledge gracefully
        overview = analyzer.generate_business_overview("HFT business")
        
        assert overview is not None
        assert isinstance(overview, dict)

class TestHFTBusinessPerformance:
    """Test performance characteristics"""
    
    def test_business_analysis_speed(self):
        """Verify business analysis is fast enough for interactive use"""
        import time
        
        analyzer = HFTBusinessAnalyzer()
        
        start_time = time.time()
        
        # Perform multiple business queries
        queries = [
            "HFT revenue models",
            "market microstructure analysis", 
            "latency optimization strategies",
            "regulatory compliance requirements",
            "technology infrastructure costs"
        ]
        
        for query in queries:
            analyzer.analyze_business_concept(query)
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        # Should complete 5 analyses in under 1 second (local knowledge base)
        assert analysis_time < 1.0, f"Business analysis too slow: {analysis_time}s for 5 queries"

class TestHFTBusinessIntegrationWithLocalAI:
    """Test integration with local AI system"""
    
    def test_integrates_with_offline_ai_system(self):
        """Test integration with local offline AI for enhanced analysis"""
        from claude_code_offline_mode import ClaudeCodeOfflineMode
        
        # Create offline system
        offline_mode = ClaudeCodeOfflineMode()
        
        # Request HFT business analysis through offline system
        result = offline_mode.handle_request("overview of the business of high-frequency trading")
        
        # Verify successful processing
        assert result["success"] == True
        assert "content" in result
        
        # Verify HFT business content
        content = result["content"].lower()
        hft_terms = ["frequency", "trading", "market", "latency", "business"]
        found_terms = [term for term in hft_terms if term in content]
        
        assert len(found_terms) >= 3, f"Should find HFT terms in offline response: {found_terms}"

if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short"])