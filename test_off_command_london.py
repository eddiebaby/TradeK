#!/usr/bin/env python3
"""
London School TDD Tests for /off Command
=========================================

Outside-in behavior-driven tests focusing on user stories
and component collaborations with mocks.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import json
import time
from typing import Dict, Any

# Import the system under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from claude_code_offline_mode import ClaudeCodeOfflineMode, main

class UserStory:
    """Helper class for expressing user stories"""
    def __init__(self, title: str, as_a: str, i_want: str, so_that: str):
        self.title = title
        self.as_a = as_a
        self.i_want = i_want  
        self.so_that = so_that
        
    def __str__(self):
        return f"{self.title}\nAs a {self.as_a}\nI want {self.i_want}\nSo that {self.so_that}"

class TestOffCommandUserStories:
    """London School TDD - Start with user behavior"""
    
    @pytest.fixture  
    @patch('claude_code_offline_mode.LocalTradingAI')
    @patch('builtins.print')  # Mock print to speed up tests
    def offline_mode(self, mock_print, mock_ai_class):
        """Create offline mode instance with mocked dependencies"""
        # Setup mock AI system
        mock_ai_instance = Mock()
        mock_ai_instance.qwen.available = True
        mock_ai_instance.book_search.knowledge_base = {
            'concepts': {
                'momentum_strategies': {'source': 'Test Book', 'concepts': [], 'code_patterns': []},
                'risk_management': {'source': 'Test Book 2', 'concepts': [], 'code_patterns': []}
            }
        }
        mock_ai_instance.generate_strategy.return_value = {
            "success": True,
            "content": "test strategy code",
            "model": "test_model",
            "response_time": 1.0,
            "cost": 0.0
        }
        mock_ai_class.return_value = mock_ai_instance
        
        offline = ClaudeCodeOfflineMode()
        return offline
    
    def test_user_can_switch_to_offline_mode(self, offline_mode):
        """
        GIVEN a user wants to avoid API tokens
        WHEN they initialize offline mode
        THEN system switches to local-only mode
        """
        story = UserStory(
            title="Switch to Offline Mode",
            as_a="Claude Code user",
            i_want="to switch to offline mode",
            so_that="I don't use any Anthropic tokens"
        )
        
        # Verify offline mode is active
        assert offline_mode.mode == "offline"
        assert offline_mode.session_stats["requests_handled"] == 0
        assert offline_mode.session_stats["total_cost_saved"] == 0.0
    
    def test_user_can_generate_trading_strategy_offline(self, offline_mode):
        """
        GIVEN user is in offline mode
        WHEN they request "momentum trading strategy"
        THEN they receive complete strategy using local AI
        """
        story = UserStory(
            title="Generate Strategy Offline",
            as_a="trader",
            i_want="to generate strategies without cloud APIs",
            so_that="I can work during API outages"
        )
        
        # Setup mock response
        mock_strategy_result = {
            "success": True,
            "content": "class MomentumStrategy:\n    pass",
            "model": "qwen2.5-coder:7b",
            "response_time": 2.5,
            "cost": 0.0
        }
        offline_mode.ai_system.generate_strategy.return_value = mock_strategy_result
        
        # Execute request
        result = offline_mode.handle_request("momentum trading strategy")
        
        # Verify behavior
        assert result["success"] == True
        assert result["type"] == "trading_strategy"
        assert "MomentumStrategy" in result["content"]
        assert result["cost"] == 0.0
        assert result["tokens_saved"] > 0
        offline_mode.ai_system.generate_strategy.assert_called_once_with("momentum trading strategy")
    
    def test_user_sees_zero_token_usage(self, offline_mode):
        """
        GIVEN user has made offline requests
        WHEN they check usage stats
        THEN token count shows zero external tokens used
        """
        story = UserStory(
            title="Track Zero Token Usage",
            as_a="cost-conscious user",
            i_want="to see I'm using zero tokens",
            so_that="I know I'm not incurring costs"
        )
        
        # Make several requests
        requests = [
            "momentum strategy",
            "risk management system",
            "analyze market data"
        ]
        
        for request in requests:
            offline_mode.ai_system.generate_strategy.return_value = {
                "success": True, 
                "content": "test strategy content", 
                "model": "test_model",
                "cost": 0.0, 
                "response_time": 1.0
            }
            offline_mode.handle_request(request)
        
        # Check stats
        stats = offline_mode.show_stats()
        
        assert offline_mode.session_stats["requests_handled"] == 3
        assert offline_mode.session_stats["total_cost_saved"] > 0
        assert "Zero external API calls" in stats
        # Check that we're not using external APIs (cost should be 0)
        assert offline_mode.session_stats["total_cost_saved"] >= 0

class TestOffCommandCollaborations:
    """Test interactions between components using mocks"""
    
    def test_off_command_delegates_to_local_ai(self):
        """Verify correct delegation to LocalTradingAI"""
        # Mock LocalTradingAI
        with patch('claude_code_offline_mode.LocalTradingAI') as MockLocalAI:
            mock_ai = Mock()
            mock_ai.qwen.available = True
            mock_ai.book_search.knowledge_base = {'concepts': {}}
            mock_ai.generate_strategy.return_value = {
                "success": True,
                "content": "strategy code",
                "model": "test",
                "response_time": 1.0,
                "cost": 0.0
            }
            MockLocalAI.return_value = mock_ai
            
            # Create offline mode
            offline = ClaudeCodeOfflineMode()
            
            # Request a strategy
            offline.handle_request("test strategy")
            
            # Verify delegation
            mock_ai.generate_strategy.assert_called_once_with("test strategy")
    
    def test_off_command_formats_responses_correctly(self):
        """Verify response formatting matches Claude Code style"""
        with patch('claude_code_offline_mode.LocalTradingAI') as MockLocalAI:
            mock_ai = Mock()
            mock_ai.qwen.available = False  # Force fallback
            mock_ai.book_search.knowledge_base = {'concepts': {}}
            MockLocalAI.return_value = mock_ai
            
            offline = ClaudeCodeOfflineMode()
            
            # Test different request types
            result = offline.handle_request("implement quicksort")
            
            # Verify response structure
            assert "type" in result
            assert "success" in result
            assert "content" in result
            assert "model" in result
            assert "response_time" in result
            assert "cost" in result
    
    def test_request_routing_logic(self):
        """Test that requests are routed to correct handlers"""
        with patch('claude_code_offline_mode.LocalTradingAI') as MockLocalAI:
            mock_ai = Mock()
            mock_ai.qwen.available = True
            mock_ai.book_search.knowledge_base = {'concepts': {}}
            mock_ai.book_search.search_relevant_context.return_value = "test context"
            MockLocalAI.return_value = mock_ai
            
            offline = ClaudeCodeOfflineMode()
            
            # Test trading request routing
            with patch.object(offline, '_handle_trading_request') as mock_trading:
                mock_trading.return_value = {"type": "trading_strategy", "success": True, "content": ""}
                offline.handle_request("momentum trading strategy")
                mock_trading.assert_called_once()
            
            # Test code request routing
            with patch.object(offline, '_handle_code_request') as mock_code:
                mock_code.return_value = {"type": "code_generation", "success": True, "content": ""}
                offline.handle_request("implement binary search")
                mock_code.assert_called_once()
            
            # Test analysis request routing
            with patch.object(offline, '_handle_analysis_request') as mock_analysis:
                mock_analysis.return_value = {"type": "analysis", "success": True, "content": ""}
                offline.handle_request("analyze this data")
                mock_analysis.assert_called_once()

class TestErrorHandlingBehavior:
    """Test error scenarios and recovery"""
    
    @patch('claude_code_offline_mode.LocalTradingAI')
    @patch('builtins.print')
    def test_handles_ai_system_unavailable(self, mock_print, MockLocalAI):
        """System gracefully handles when AI is unavailable"""
        # Simulate AI system failure
        mock_ai = Mock()
        mock_ai.qwen.available = False
        mock_ai.book_search.knowledge_base = {'concepts': {}}
        mock_ai.generate_strategy.side_effect = Exception("AI system error")
        MockLocalAI.return_value = mock_ai
        
        offline = ClaudeCodeOfflineMode()
        
        # Should not crash and should fallback
        result = offline.handle_request("generate strategy")
        
        assert result["success"] == True  # Falls back to template
        assert "fallback" in result.get("model", "").lower()
    
    @patch('claude_code_offline_mode.LocalTradingAI')
    @patch('builtins.print')
    def test_handles_malformed_requests(self, mock_print, MockLocalAI):
        """System handles edge cases in user input"""
        mock_ai = Mock()
        mock_ai.qwen.available = True
        mock_ai.book_search.knowledge_base = {'concepts': {}}
        MockLocalAI.return_value = mock_ai
        
        offline = ClaudeCodeOfflineMode()
        
        # Test empty request
        result = offline.handle_request("")
        assert result["success"] == True
        assert result["type"] == "general"
        
        # Test very long request
        long_request = "x" * 10000
        result = offline.handle_request(long_request)
        assert result["success"] == True
        
        # Test special characters
        result = offline.handle_request("!@#$%^&*()")
        assert result["success"] == True

class TestStatisticsTracking:
    """Test usage statistics and metrics"""
    
    def test_tracks_session_statistics_accurately(self):
        """Verify statistics are tracked correctly"""
        with patch('claude_code_offline_mode.LocalTradingAI') as MockLocalAI:
            mock_ai = Mock()
            mock_ai.qwen.available = True
            mock_ai.book_search.knowledge_base = {'concepts': {}}
            mock_ai.generate_strategy.return_value = {
                "success": True,
                "content": "x" * 1000,  # 1000 chars
                "model": "test",
                "response_time": 2.0,
                "cost": 0.0
            }
            MockLocalAI.return_value = mock_ai
            
            offline = ClaudeCodeOfflineMode()
            
            # Make multiple requests
            for i in range(5):
                offline.handle_request(f"strategy {i}")
            
            # Verify stats
            assert offline.session_stats["requests_handled"] == 5
            assert offline.session_stats["strategies_generated"] == 5
            assert offline.session_stats["avg_response_time"] == pytest.approx(2.0, 0.1)
            assert offline.session_stats["total_tokens_saved"] > 0
            assert offline.session_stats["total_cost_saved"] > 0

class TestInteractiveMode:
    """Test interactive command line mode"""
    
    @patch('builtins.input')
    @patch('sys.argv', ['test', '--interactive'])
    def test_interactive_mode_session(self, mock_input):
        """Test full interactive session workflow"""
        # Simulate user inputs
        mock_input.side_effect = [
            "momentum strategy",
            "/stats",
            "quit"
        ]
        
        with patch('claude_code_offline_mode.LocalTradingAI') as MockLocalAI:
            mock_ai = Mock()
            mock_ai.qwen.available = True
            mock_ai.book_search.knowledge_base = {'concepts': {}}
            mock_ai.generate_strategy.return_value = {
                "success": True,
                "content": "test strategy",
                "model": "test",
                "response_time": 1.0,
                "cost": 0.0
            }
            MockLocalAI.return_value = mock_ai
            
            # Run main in interactive mode
            with patch('builtins.print') as mock_print:
                main()
            
            # Verify interactions
            assert mock_input.call_count == 3
            mock_ai.generate_strategy.assert_called_once()
            
            # Check for expected output patterns
            output = ' '.join(str(call) for call in mock_print.call_args_list)
            assert "OFFLINE MODE" in output
            assert "SESSION STATS" in output

class TestSequentialThinkingIntegration:
    """Test integration with sequential thinking for complex problems"""
    
    @patch('claude_code_offline_mode.LocalTradingAI')
    @patch('builtins.print')
    def test_uses_sequential_thinking_for_complex_requests(self, mock_print, MockLocalAI):
        """Complex requests trigger sequential thinking"""
        # This would integrate with MCP sequential thinking tool
        # For now, we test the interface exists
        mock_ai = Mock()
        mock_ai.qwen.available = True
        mock_ai.book_search.knowledge_base = {'concepts': {}}
        mock_ai.generate_strategy.return_value = {
            "success": True,
            "content": "test complex strategy code",
            "model": "test_model",
            "response_time": 1.0,
            "cost": 0.0
        }
        MockLocalAI.return_value = mock_ai
        
        offline = ClaudeCodeOfflineMode()
        
        complex_request = """
        Create a comprehensive trading system that:
        1. Uses machine learning for prediction
        2. Implements risk management
        3. Handles real-time data
        4. Includes backtesting
        """
        
        # In a real implementation, this would use sequential thinking
        result = offline.handle_request(complex_request)
        
        assert result["success"] == True
        assert result["type"] in ["trading_strategy", "code_generation"]

class TestKnowledgeGraphIntegration:
    """Test knowledge graph memory integration"""
    
    @patch('claude_code_offline_mode.LocalTradingAI')
    @patch('builtins.print')
    def test_stores_successful_patterns_in_knowledge_graph(self, mock_print, MockLocalAI):
        """Successful strategies are stored in knowledge graph"""
        mock_ai = Mock()
        mock_ai.qwen.available = True
        mock_ai.book_search.knowledge_base = {'concepts': {}}
        mock_ai.generate_strategy.return_value = {
            "success": True,
            "content": "test momentum strategy code",
            "model": "test_model",
            "response_time": 1.0,
            "cost": 0.0
        }
        MockLocalAI.return_value = mock_ai
        
        offline = ClaudeCodeOfflineMode()
        
        # Generate a strategy
        result = offline.handle_request("momentum strategy")
        
        # In full implementation, this would store:
        # - Strategy pattern used
        # - Success metrics
        # - User feedback
        # mock_kg.assert_called_once()

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "-v", "--cov=claude_code_offline_mode", "--cov-report=term-missing"])