"""
Query suggestion engine for TradeKnowledge

This provides intelligent query suggestions based on:
- Previous successful searches
- Common patterns in the corpus
- Spelling corrections
- Related terms
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import asyncio
import re
from datetime import datetime, timedelta

import numpy as np
try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None

logger = logging.getLogger(__name__)

class QuerySuggester:
    """
    Provides intelligent query suggestions for better search experience.
    
    Features:
    - Autocomplete from search history
    - Spelling correction
    - Synonym suggestions
    - Related terms from corpus
    - Query expansion for trading terms
    """
    
    def __init__(self):
        """Initialize query suggester"""
        self.storage = None
        self.cache_manager = None
        
        # Spell checker (if available)
        self.spell_checker = None
        if SpellChecker:
            self.spell_checker = SpellChecker()
        
        # Trading-specific terms to add to dictionary
        self.trading_terms = {
            'sma', 'ema', 'macd', 'rsi', 'bollinger', 'ichimoku',
            'backtest', 'sharpe', 'sortino', 'drawdown', 'slippage',
            'arbitrage', 'hedging', 'derivatives', 'futures', 'options',
            'forex', 'cryptocurrency', 'bitcoin', 'ethereum', 'defi',
            'quantitative', 'algorithmic', 'hft', 'market-making',
            'mean-reversion', 'momentum', 'breakout', 'scalping'
        }
        
        # Add trading terms to spell checker if available
        if self.spell_checker:
            self.spell_checker.word_frequency.load_words(self.trading_terms)
        
        # Query patterns for trading
        self.query_patterns = {
            'strategy': re.compile(r'(\w+)\s+(?:strategy|strategies|system)', re.I),
            'indicator': re.compile(r'(\w+)\s+(?:indicator|signal|oscillator)', re.I),
            'code': re.compile(r'(?:python|code|implement|example)\s+(\w+)', re.I),
            'formula': re.compile(r'(?:formula|equation|calculate)\s+(\w+)', re.I)
        }
        
        # Common query templates
        self.templates = {
            'how_to': "how to {topic}",
            'what_is': "what is {topic}",
            'python_code': "python code for {topic}",
            'example': "{topic} example",
            'tutorial': "{topic} tutorial",
            'vs': "{topic1} vs {topic2}",
            'best': "best {topic} strategy"
        }
        
        # Term relationships for expansion
        self.related_terms = {
            'moving average': ['sma', 'ema', 'wma', 'trend following'],
            'momentum': ['rsi', 'macd', 'stochastic', 'rate of change'],
            'volatility': ['atr', 'bollinger bands', 'standard deviation', 'vix'],
            'risk': ['var', 'cvar', 'sharpe ratio', 'risk management'],
            'backtest': ['historical data', 'simulation', 'performance metrics'],
            'portfolio': ['diversification', 'allocation', 'optimization', 'rebalancing']
        }
        
        # Search history for suggestions
        self.search_history = Counter()
        self.successful_queries = set()
    
    async def initialize(self):
        """Initialize components"""
        try:
            from core.sqlite_storage import SQLiteStorage
            from utils.cache_manager import get_cache_manager
            
            self.storage = SQLiteStorage()
            self.cache_manager = await get_cache_manager()
            
            # Load search history if available
            await self._load_search_history()
            
        except ImportError as e:
            logger.warning(f"Some components not available: {e}")
    
    async def suggest(self, 
                     partial_query: str,
                     max_suggestions: int = 10) -> List[Dict[str, Any]]:
        """
        Get query suggestions for partial input.
        
        Args:
            partial_query: Partial query string
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggestions with metadata
        """
        if not partial_query or len(partial_query) < 2:
            return []
        
        suggestions = []
        partial_lower = partial_query.lower().strip()
        
        # 1. Autocomplete from search history
        history_suggestions = await self._get_history_suggestions(
            partial_lower, max_suggestions
        )
        suggestions.extend(history_suggestions)
        
        # 2. Spelling corrections
        if len(partial_query.split()) <= 3 and self.spell_checker:
            spell_suggestions = await self._get_spelling_suggestions(partial_query)
            suggestions.extend(spell_suggestions)
        
        # 3. Template-based suggestions
        template_suggestions = self._get_template_suggestions(partial_lower)
        suggestions.extend(template_suggestions)
        
        # 4. Related term suggestions
        related_suggestions = self._get_related_suggestions(partial_lower)
        suggestions.extend(related_suggestions)
        
        # Deduplicate and rank
        unique_suggestions = self._deduplicate_suggestions(suggestions)
        ranked_suggestions = self._rank_suggestions(unique_suggestions, partial_lower)
        
        return ranked_suggestions[:max_suggestions]
    
    async def _get_history_suggestions(self, 
                                     partial: str,
                                     limit: int) -> List[Dict[str, Any]]:
        """Get suggestions from search history"""
        suggestions = []
        
        for query, count in self.search_history.most_common():
            if query.lower().startswith(partial):
                suggestions.append({
                    'text': query,
                    'type': 'history',
                    'score': count * 0.1,  # Boost popular queries
                    'metadata': {'usage_count': count}
                })
                
                if len(suggestions) >= limit:
                    break
        
        return suggestions
    
    async def _get_spelling_suggestions(self, query: str) -> List[Dict[str, Any]]:
        """Get spelling correction suggestions"""
        if not self.spell_checker:
            return []
        
        suggestions = []
        words = query.split()
        
        for i, word in enumerate(words):
            if word.lower() not in self.spell_checker:
                # Get corrections for this word
                corrections = self.spell_checker.candidates(word)
                
                for correction in list(corrections)[:3]:  # Top 3 corrections
                    corrected_words = words.copy()
                    corrected_words[i] = correction
                    corrected_query = ' '.join(corrected_words)
                    
                    suggestions.append({
                        'text': corrected_query,
                        'type': 'spelling',
                        'score': 0.8,
                        'metadata': {
                            'original': query,
                            'corrected_word': correction,
                            'position': i
                        }
                    })
        
        return suggestions
    
    def _get_template_suggestions(self, partial: str) -> List[Dict[str, Any]]:
        """Get template-based suggestions"""
        suggestions = []
        
        # Check if partial matches any template patterns
        for template_name, template in self.templates.items():
            # Simple template matching for demonstration
            if any(word in partial for word in ['how', 'what', 'python', 'example']):
                # Extract potential topic from partial
                words = partial.split()
                if len(words) >= 2:
                    topic = words[-1]  # Use last word as topic
                    
                    if topic in self.trading_terms:
                        filled_template = template.format(topic=topic)
                        suggestions.append({
                            'text': filled_template,
                            'type': 'template',
                            'score': 0.6,
                            'metadata': {
                                'template': template_name,
                                'topic': topic
                            }
                        })
        
        return suggestions
    
    def _get_related_suggestions(self, partial: str) -> List[Dict[str, Any]]:
        """Get suggestions based on related terms"""
        suggestions = []
        
        # Find related terms
        for base_term, related in self.related_terms.items():
            if base_term in partial or any(term in partial for term in related):
                for related_term in related:
                    if related_term not in partial:
                        # Suggest query with related term
                        new_query = f"{partial} {related_term}".strip()
                        
                        suggestions.append({
                            'text': new_query,
                            'type': 'related',
                            'score': 0.5,
                            'metadata': {
                                'base_term': base_term,
                                'related_term': related_term
                            }
                        })
        
        return suggestions
    
    def _deduplicate_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate suggestions"""
        seen = set()
        unique = []
        
        for suggestion in suggestions:
            text = suggestion['text'].lower()
            if text not in seen:
                seen.add(text)
                unique.append(suggestion)
        
        return unique
    
    def _rank_suggestions(self, 
                         suggestions: List[Dict[str, Any]],
                         original_query: str) -> List[Dict[str, Any]]:
        """Rank suggestions by relevance"""
        # Calculate relevance scores
        for suggestion in suggestions:
            text = suggestion['text']
            base_score = suggestion['score']
            
            # Boost exact prefix matches
            if text.lower().startswith(original_query):
                base_score += 1.0
            
            # Boost shorter suggestions (more specific)
            length_penalty = len(text.split()) * 0.05
            base_score -= length_penalty
            
            # Type-based bonuses
            type_bonuses = {
                'history': 0.8,
                'spelling': 0.6,
                'template': 0.4,
                'related': 0.3
            }
            base_score += type_bonuses.get(suggestion['type'], 0)
            
            suggestion['final_score'] = base_score
        
        # Sort by final score
        return sorted(suggestions, key=lambda x: x['final_score'], reverse=True)
    
    async def record_search(self, query: str, result_count: int = 0):
        """Record a search query for future suggestions"""
        if not query or len(query) < 3:
            return
        
        # Add to history
        self.search_history[query] += 1
        
        # Mark as successful if it returned results
        if result_count > 0:
            self.successful_queries.add(query)
        
        # Save to storage if available
        if self.storage:
            try:
                # Store in a simple format for now
                await self._save_search_history()
            except Exception as e:
                logger.error(f"Failed to save search history: {e}")
    
    async def _load_search_history(self):
        """Load search history from storage"""
        if not self.storage:
            return
        
        try:
            # Simple implementation - in real system, use proper database table
            history_data = await self.cache_manager.get("search_history", "general")
            if history_data:
                self.search_history.update(history_data)
        except Exception as e:
            logger.error(f"Failed to load search history: {e}")
    
    async def _save_search_history(self):
        """Save search history to storage"""
        if not self.cache_manager:
            return
        
        try:
            # Keep only top 1000 queries to prevent unlimited growth
            top_queries = dict(self.search_history.most_common(1000))
            await self.cache_manager.set("search_history", top_queries, "general", ttl=86400*7)  # 1 week
        except Exception as e:
            logger.error(f"Failed to save search history: {e}")
    
    def get_popular_queries(self, limit: int = 10) -> List[str]:
        """Get most popular search queries"""
        return [query for query, _ in self.search_history.most_common(limit)]
    
    def get_trending_terms(self, days: int = 7) -> List[str]:
        """Get trending search terms (simplified implementation)"""
        # In a real implementation, this would analyze time-based trends
        return self.get_popular_queries(20)
    
    async def expand_query(self, query: str) -> List[str]:
        """Expand query with related terms"""
        expanded = [query]
        query_lower = query.lower()
        
        # Add related terms
        for base_term, related in self.related_terms.items():
            if base_term in query_lower:
                for related_term in related[:3]:  # Top 3 related terms
                    expanded.append(f"{query} {related_term}")
        
        return expanded
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.cache_manager:
            await self._save_search_history()


# Example usage and testing
async def test_query_suggester():
    """Test the query suggester"""
    suggester = QuerySuggester()
    await suggester.initialize()
    
    # Simulate some search history
    await suggester.record_search("moving average strategy", 5)
    await suggester.record_search("rsi indicator", 3)
    await suggester.record_search("python backtest", 10)
    await suggester.record_search("sharpe ratio calculation", 2)
    
    # Test suggestions
    test_queries = [
        "mov",
        "rsi",
        "python",
        "what is",
        "how to",
        "risk"
    ]
    
    for query in test_queries:
        suggestions = await suggester.suggest(query, max_suggestions=5)
        
        print(f"\nSuggestions for '{query}':")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion['text']} ({suggestion['type']}, score: {suggestion['final_score']:.2f})")
    
    # Test query expansion
    expanded = await suggester.expand_query("momentum strategy")
    print(f"\nExpanded 'momentum strategy': {expanded}")
    
    # Show popular queries
    popular = suggester.get_popular_queries(5)
    print(f"\nPopular queries: {popular}")


if __name__ == "__main__":
    asyncio.run(test_query_suggester())