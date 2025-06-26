"""
Enhanced RESEARCHER Agent with OpenAI Tools Integration

This module enhances the existing RESEARCHER agent with OpenAI agents SDK tools,
specifically WebSearchTool for real-time market intelligence gathering.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from agents import Agent, WebSearchTool, Runner, trace
from agents.researcher.researcher_agent import (
    ResearcherAgent, 
    ResearchRequest, 
    ResearchIntelligence, 
    ResearchInsight,
    ResearchDomain
)
from core.agent_base import TaskContext, AgentRole


@dataclass
class EnhancedResearchResult:
    """Enhanced research result combining traditional and web search intelligence."""
    traditional_research: ResearchIntelligence
    web_intelligence: Dict[str, Any]
    market_trends: List[Dict[str, Any]]
    real_time_insights: List[Dict[str, Any]]
    synthesis_quality: float
    research_timestamp: float = field(default_factory=time.time)


class EnhancedResearcherAgent(ResearcherAgent):
    """
    Enhanced RESEARCHER Agent with OpenAI Tools Integration
    
    Combines the existing comprehensive research capabilities with:
    - Real-time web search intelligence
    - Market trend analysis
    - Live data integration
    - Enhanced insight synthesis
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        super().__init__()
        
        # Initialize OpenAI Agent with WebSearchTool
        self.web_agent = Agent(
            name="TradeKnowledge-WebSearcher",
            instructions="""
            You are a specialized web research agent for financial and trading knowledge.
            
            Focus on:
            - Market trends and financial news
            - Trading strategies and methodologies
            - Technology developments in fintech
            - Regulatory changes and compliance updates
            - Industry best practices and benchmarks
            
            Provide concise, actionable insights with confidence scores.
            """,
            tools=[
                WebSearchTool(
                    user_location={"type": "approximate", "city": "New York"},
                    max_results=10
                )
            ]
        )
        
        # Enhanced capabilities
        self.enhanced_capabilities = [
            "real_time_web_search",
            "market_intelligence_gathering",
            "trend_analysis_live",
            "news_sentiment_analysis",
            "competitive_landscape_monitoring",
            "regulatory_update_tracking",
            "technology_trend_identification",
            "expert_opinion_synthesis"
        ]
        
        # Web search specializations
        self.web_search_domains = {
            "market_intelligence": {
                "keywords": ["market trends", "trading strategies", "financial news"],
                "sources": ["Bloomberg", "Reuters", "Financial Times", "Wall Street Journal"]
            },
            "technology_trends": {
                "keywords": ["fintech", "algorithmic trading", "AI finance", "blockchain"],
                "sources": ["TechCrunch", "Forbes Tech", "MIT Technology Review"]
            },
            "regulatory_updates": {
                "keywords": ["SEC regulations", "financial compliance", "trading rules"],
                "sources": ["SEC.gov", "FINRA", "regulatory announcements"]
            },
            "competitive_analysis": {
                "keywords": ["hedge funds", "trading platforms", "robo advisors"],
                "sources": ["industry reports", "competitor analysis", "market share"]
            }
        }
        
        # Quality thresholds for web intelligence
        self.web_intelligence_thresholds = {
            "relevance_score": 0.75,
            "confidence_threshold": 0.80,
            "source_credibility": 0.85,
            "recency_weight": 0.90  # Prefer recent information
        }
    
    async def conduct_enhanced_research(self, research_spec: Dict[str, Any]) -> EnhancedResearchResult:
        """
        Conduct enhanced research combining traditional methods with real-time web search.
        
        Args:
            research_spec: Research specification with domains and requirements
            
        Returns:
            EnhancedResearchResult: Comprehensive research with web intelligence
        """
        research_start = time.time()
        
        # Execute traditional research in parallel with web search
        traditional_task = asyncio.create_task(
            self.conduct_comprehensive_research(research_spec)
        )
        
        web_intelligence_task = asyncio.create_task(
            self._conduct_web_intelligence_gathering(research_spec)
        )
        
        # Wait for both research streams to complete
        traditional_research, web_intelligence = await asyncio.gather(
            traditional_task, web_intelligence_task
        )
        
        # Extract market trends and real-time insights
        market_trends = await self._extract_market_trends(web_intelligence)
        real_time_insights = await self._extract_real_time_insights(web_intelligence)
        
        # Synthesize combined intelligence
        synthesis_quality = await self._calculate_synthesis_quality(
            traditional_research, web_intelligence
        )
        
        return EnhancedResearchResult(
            traditional_research=traditional_research,
            web_intelligence=web_intelligence,
            market_trends=market_trends,
            real_time_insights=real_time_insights,
            synthesis_quality=synthesis_quality,
            research_timestamp=time.time()
        )
    
    async def _conduct_web_intelligence_gathering(self, research_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conduct real-time web intelligence gathering using OpenAI WebSearchTool.
        
        Args:
            research_spec: Research specification
            
        Returns:
            Dict containing web intelligence results
        """
        web_intelligence = {
            "search_results": [],
            "market_insights": [],
            "trend_analysis": {},
            "news_sentiment": {},
            "competitive_intelligence": [],
            "regulatory_updates": [],
            "expert_opinions": []
        }
        
        # Determine web search strategy based on research domains
        search_queries = self._generate_search_queries(research_spec)
        
        for query_info in search_queries:
            try:
                with trace(f"Web search: {query_info['query']}"):
                    # Execute web search using OpenAI Agent
                    search_prompt = self._build_search_prompt(query_info)
                    result = await Runner.run(
                        starting_agent=self.web_agent,
                        input=search_prompt
                    )
                    
                    # Process and categorize results
                    processed_result = await self._process_web_search_result(
                        result, query_info
                    )
                    
                    # Add to appropriate category
                    category = query_info.get("category", "general")
                    if category == "market_intelligence":
                        web_intelligence["market_insights"].append(processed_result)
                    elif category == "regulatory":
                        web_intelligence["regulatory_updates"].append(processed_result)
                    elif category == "competitive":
                        web_intelligence["competitive_intelligence"].append(processed_result)
                    else:
                        web_intelligence["search_results"].append(processed_result)
                    
            except Exception as e:
                self.logger.warning(f"Web search failed for query {query_info['query']}: {e}")
                continue
        
        # Analyze sentiment and trends
        web_intelligence["news_sentiment"] = await self._analyze_news_sentiment(
            web_intelligence["search_results"]
        )
        web_intelligence["trend_analysis"] = await self._analyze_web_trends(
            web_intelligence["market_insights"]
        )
        
        return web_intelligence
    
    def _generate_search_queries(self, research_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate targeted search queries based on research specification."""
        
        queries = []
        focus_areas = research_spec.get("focus_areas", ["general"])
        domains = research_spec.get("domains", ["technical_deep_dive"])
        
        # Base queries for financial/trading research
        base_queries = [
            {
                "query": "latest trading strategies 2024 algorithmic trading",
                "category": "market_intelligence",
                "priority": "high"
            },
            {
                "query": "financial markets technology trends AI machine learning",
                "category": "technology",
                "priority": "high"
            },
            {
                "query": "SEC regulations trading compliance 2024 updates",
                "category": "regulatory",
                "priority": "medium"
            }
        ]
        
        # Add domain-specific queries
        for domain in domains:
            if domain == "market_intelligence":
                queries.extend([
                    {
                        "query": "hedge fund strategies quantitative trading 2024",
                        "category": "market_intelligence",
                        "priority": "high"
                    },
                    {
                        "query": "market volatility analysis trading signals",
                        "category": "market_intelligence",
                        "priority": "medium"
                    }
                ])
            elif domain == "technical_analysis":
                queries.extend([
                    {
                        "query": "Python trading algorithms FastAPI financial services",
                        "category": "technology",
                        "priority": "high"
                    },
                    {
                        "query": "vector databases financial data Qdrant trading systems",
                        "category": "technology",
                        "priority": "medium"
                    }
                ])
        
        # Add focus area specific queries
        for focus_area in focus_areas:
            if "performance" in focus_area.lower():
                queries.append({
                    "query": f"high performance trading systems {focus_area} optimization",
                    "category": "performance",
                    "priority": "high"
                })
            elif "security" in focus_area.lower():
                queries.append({
                    "query": f"financial security {focus_area} cybersecurity trading",
                    "category": "security",
                    "priority": "high"
                })
        
        return base_queries + queries
    
    def _build_search_prompt(self, query_info: Dict[str, Any]) -> str:
        """Build optimized search prompt for the web agent."""
        
        query = query_info["query"]
        category = query_info.get("category", "general")
        priority = query_info.get("priority", "medium")
        
        prompt = f"""
        Search for: "{query}"
        
        Focus on {category} information with {priority} priority.
        
        Please provide:
        1. Top 3-5 most relevant and recent results
        2. Key insights with confidence scores (0-1)
        3. Source credibility assessment
        4. Actionable recommendations
        5. Risk factors or considerations
        
        Prioritize:
        - Recent information (last 6 months preferred)
        - Authoritative sources
        - Actionable insights
        - Quantitative data where available
        
        Format as structured insights with clear source attribution.
        """
        
        return prompt
    
    async def _process_web_search_result(self, result: Any, query_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure web search results."""
        
        return {
            "query": query_info["query"],
            "category": query_info.get("category", "general"),
            "priority": query_info.get("priority", "medium"),
            "content": result.final_output,
            "insights": await self._extract_insights_from_content(result.final_output),
            "confidence_score": self._calculate_content_confidence(result.final_output),
            "relevance_score": self._calculate_content_relevance(result.final_output, query_info),
            "timestamp": time.time(),
            "source_quality": "web_search"
        }
    
    async def _extract_insights_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Extract structured insights from web search content."""
        
        # This would typically use NLP processing, for now simulate structured extraction
        insights = []
        
        # Look for key patterns in content
        if "trend" in content.lower():
            insights.append({
                "type": "trend",
                "description": "Market trend identified",
                "confidence": 0.8
            })
        
        if "regulation" in content.lower() or "compliance" in content.lower():
            insights.append({
                "type": "regulatory",
                "description": "Regulatory consideration identified",
                "confidence": 0.85
            })
        
        if "performance" in content.lower() or "optimization" in content.lower():
            insights.append({
                "type": "performance",
                "description": "Performance optimization opportunity",
                "confidence": 0.75
            })
        
        return insights
    
    def _calculate_content_confidence(self, content: str) -> float:
        """Calculate confidence score for web search content."""
        
        # Simple heuristic-based confidence calculation
        confidence = 0.5  # Base confidence
        
        # Increase confidence for longer, more detailed content
        if len(content) > 500:
            confidence += 0.2
        
        # Increase confidence for structured content
        if any(marker in content for marker in ["1.", "2.", "3.", "-", "*"]):
            confidence += 0.1
        
        # Increase confidence for specific data/numbers
        import re
        if re.search(r'\d+%|\$\d+|\d+\.\d+', content):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_content_relevance(self, content: str, query_info: Dict[str, Any]) -> float:
        """Calculate relevance score between content and original query."""
        
        query_keywords = query_info["query"].lower().split()
        content_lower = content.lower()
        
        # Count keyword matches
        matches = sum(1 for keyword in query_keywords if keyword in content_lower)
        relevance = matches / len(query_keywords) if query_keywords else 0
        
        # Boost for category-specific terms
        category = query_info.get("category", "general")
        if category == "market_intelligence" and any(
            term in content_lower for term in ["market", "trading", "financial", "strategy"]
        ):
            relevance += 0.2
        
        return min(relevance, 1.0)
    
    async def _extract_market_trends(self, web_intelligence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract market trends from web intelligence data."""
        
        trends = []
        
        for insight in web_intelligence.get("market_insights", []):
            if insight.get("confidence_score", 0) >= self.web_intelligence_thresholds["confidence_threshold"]:
                trend = {
                    "trend_type": "market_movement",
                    "description": insight.get("content", "")[:200] + "...",
                    "confidence": insight.get("confidence_score", 0),
                    "source": "web_search",
                    "timestamp": insight.get("timestamp", time.time()),
                    "actionable_insights": insight.get("insights", [])
                }
                trends.append(trend)
        
        return trends
    
    async def _extract_real_time_insights(self, web_intelligence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract real-time actionable insights from web intelligence."""
        
        insights = []
        
        for category in ["market_insights", "regulatory_updates", "competitive_intelligence"]:
            for item in web_intelligence.get(category, []):
                if item.get("relevance_score", 0) >= self.web_intelligence_thresholds["relevance_score"]:
                    insight = {
                        "insight_type": category,
                        "title": f"Real-time {category.replace('_', ' ').title()}",
                        "description": item.get("content", "")[:300] + "...",
                        "confidence": item.get("confidence_score", 0),
                        "relevance": item.get("relevance_score", 0),
                        "category": category,
                        "timestamp": item.get("timestamp", time.time()),
                        "action_required": item.get("confidence_score", 0) > 0.9
                    }
                    insights.append(insight)
        
        return insights
    
    async def _analyze_news_sentiment(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment from news and web search results."""
        
        # Simple sentiment analysis based on keyword presence
        positive_keywords = ["growth", "increase", "bullish", "opportunity", "gain", "profit"]
        negative_keywords = ["decline", "loss", "bearish", "risk", "decrease", "volatility"]
        
        sentiment_scores = []
        
        for result in search_results:
            content = result.get("content", "").lower()
            positive_count = sum(1 for word in positive_keywords if word in content)
            negative_count = sum(1 for word in negative_keywords if word in content)
            
            if positive_count + negative_count > 0:
                sentiment = (positive_count - negative_count) / (positive_count + negative_count)
                sentiment_scores.append(sentiment)
        
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            sentiment_label = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
        else:
            avg_sentiment = 0.0
            sentiment_label = "neutral"
        
        return {
            "overall_sentiment": sentiment_label,
            "sentiment_score": avg_sentiment,
            "confidence": 0.7,  # Medium confidence for simple sentiment analysis
            "sample_size": len(sentiment_scores)
        }
    
    async def _analyze_web_trends(self, market_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends from web-based market insights."""
        
        trends = {
            "emerging_technologies": [],
            "market_directions": [],
            "regulatory_changes": [],
            "competitive_landscape": []
        }
        
        for insight in market_insights:
            content = insight.get("content", "").lower()
            
            # Identify technology trends
            if any(tech in content for tech in ["ai", "machine learning", "blockchain", "api"]):
                trends["emerging_technologies"].append({
                    "technology": "AI/ML in Finance",
                    "adoption_stage": "growing",
                    "confidence": insight.get("confidence_score", 0.5)
                })
            
            # Identify market directions
            if any(direction in content for direction in ["bullish", "bearish", "volatile"]):
                trends["market_directions"].append({
                    "direction": "mixed_signals",
                    "timeframe": "short_term",
                    "confidence": insight.get("confidence_score", 0.5)
                })
        
        return trends
    
    async def _calculate_synthesis_quality(self, 
                                         traditional_research: ResearchIntelligence,
                                         web_intelligence: Dict[str, Any]) -> float:
        """Calculate quality score for research synthesis."""
        
        # Factor 1: Traditional research quality
        traditional_quality = traditional_research.quality_metrics.get("confidence_average", 0.5)
        
        # Factor 2: Web intelligence quality
        web_insights = web_intelligence.get("market_insights", [])
        web_quality = (
            sum(insight.get("confidence_score", 0) for insight in web_insights) / 
            len(web_insights) if web_insights else 0.5
        )
        
        # Factor 3: Data recency (web intelligence gets higher weight for recency)
        recency_factor = 0.9  # Web data is typically more recent
        
        # Factor 4: Source diversity
        source_diversity = min(
            (len(traditional_research.insights) + len(web_insights)) / 10, 1.0
        )
        
        # Weighted synthesis quality
        synthesis_quality = (
            traditional_quality * 0.4 +
            web_quality * 0.3 +
            recency_factor * 0.2 +
            source_diversity * 0.1
        )
        
        return synthesis_quality
    
    async def get_enhanced_insights_for_handoff(self, task: TaskContext) -> Dict[str, Any]:
        """Get enhanced research insights including web intelligence for handoff."""
        
        base_insights = self.get_insights_for_handoff(task)
        
        # Add web intelligence context
        enhanced_insights = {
            **base_insights,
            "web_intelligence_available": True,
            "real_time_capabilities": self.enhanced_capabilities,
            "market_intelligence_quality": "high_confidence_web_enabled",
            "trend_analysis_scope": "traditional_plus_real_time",
            "competitive_intelligence": "live_market_monitoring",
            "research_methodology": "hybrid_traditional_web_intelligence"
        }
        
        return enhanced_insights
    
    def get_enhanced_capabilities(self) -> List[str]:
        """Return enhanced capabilities including web search integration."""
        return self.capabilities + self.enhanced_capabilities