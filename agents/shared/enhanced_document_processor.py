"""
Enhanced Document Processor with OpenAI FileSearchTool Integration

This module enhances document processing capabilities with OpenAI's FileSearchTool
for advanced semantic search and document analysis.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from agents import Agent, FileSearchTool, Runner, trace
from src.ingestion.enhanced_book_processor import EnhancedBookProcessor
from src.core.models import Document, SearchResult


@dataclass
class EnhancedDocumentResult:
    """Enhanced document analysis result with OpenAI FileSearchTool integration."""
    document_id: str
    traditional_processing: Dict[str, Any]
    semantic_search_results: List[Dict[str, Any]]
    ai_insights: List[Dict[str, Any]]
    document_intelligence: Dict[str, Any]
    processing_quality: float
    processing_timestamp: float = field(default_factory=time.time)


@dataclass
class DocumentIntelligence:
    """Document intelligence extracted via AI analysis."""
    key_concepts: List[str]
    sentiment_analysis: Dict[str, Any]
    complexity_assessment: Dict[str, Any]
    topic_classification: List[str]
    entity_extraction: List[Dict[str, Any]]
    relationship_mapping: Dict[str, Any]
    actionable_insights: List[str]
    confidence_score: float


class EnhancedDocumentProcessor:
    """
    Enhanced Document Processor with OpenAI FileSearchTool Integration
    
    Combines traditional document processing with:
    - Advanced semantic search capabilities
    - AI-powered document intelligence
    - Multi-modal document analysis
    - Contextual relationship mapping
    - Intelligent content extraction
    """
    
    def __init__(self, 
                 vector_store_ids: Optional[List[str]] = None,
                 openai_api_key: Optional[str] = None):
        
        # Initialize traditional document processor
        self.traditional_processor = EnhancedBookProcessor()
        
        # Vector store configuration for FileSearchTool
        self.vector_store_ids = vector_store_ids or []
        
        # Initialize OpenAI Agent with FileSearchTool
        self.document_agent = Agent(
            name="TradeKnowledge-DocumentAnalyzer",
            instructions="""
            You are a specialized document analysis agent for financial and trading knowledge.
            
            Focus on:
            - Financial document analysis and summarization
            - Trading strategy extraction from documents
            - Market intelligence gathering from reports
            - Technical documentation analysis
            - Research paper insights extraction
            - Regulatory document compliance analysis
            - Risk assessment from financial documents
            
            Provide structured analysis with:
            - Key concepts and themes
            - Actionable insights and recommendations
            - Sentiment and risk assessment
            - Entity extraction (companies, people, concepts)
            - Relationship mapping between concepts
            - Confidence scores for insights
            
            Always provide context-aware analysis relevant to trading and finance.
            """,
            tools=[
                FileSearchTool(
                    max_num_results=10,
                    vector_store_ids=self.vector_store_ids,
                    include_search_results=True
                )
            ] if self.vector_store_ids else []
        )
        
        # Enhanced processing capabilities
        self.enhanced_capabilities = [
            "semantic_document_search",
            "ai_powered_summarization",
            "contextual_insight_extraction",
            "multi_document_correlation",
            "entity_relationship_mapping",
            "sentiment_risk_analysis",
            "topic_classification",
            "document_intelligence_scoring"
        ]
        
        # Document analysis strategies
        self.analysis_strategies = {
            "financial_reports": {
                "focus": ["financial_metrics", "risk_factors", "market_outlook"],
                "extraction_targets": ["revenue", "profit", "forecasts", "risks"]
            },
            "research_papers": {
                "focus": ["methodology", "findings", "implications"],
                "extraction_targets": ["hypotheses", "results", "conclusions", "limitations"]
            },
            "trading_strategies": {
                "focus": ["entry_criteria", "exit_rules", "risk_management"],
                "extraction_targets": ["signals", "backtesting", "performance", "drawdowns"]
            },
            "regulatory_documents": {
                "focus": ["compliance_requirements", "deadlines", "penalties"],
                "extraction_targets": ["rules", "exceptions", "timelines", "enforcement"]
            },
            "market_analysis": {
                "focus": ["trends", "catalysts", "sentiment"],
                "extraction_targets": ["indicators", "predictions", "recommendations", "risks"]
            }
        }
        
        # Quality thresholds
        self.processing_thresholds = {
            "semantic_relevance": 0.75,
            "insight_confidence": 0.80,
            "entity_extraction_confidence": 0.70,
            "sentiment_confidence": 0.85
        }
    
    async def process_enhanced_document(self, 
                                      document_path: Union[str, Path],
                                      document_type: str = "general",
                                      analysis_depth: str = "comprehensive") -> EnhancedDocumentResult:
        """
        Process document with enhanced AI capabilities.
        
        Args:
            document_path: Path to the document to process
            document_type: Type of document for specialized analysis
            analysis_depth: Depth of analysis ("quick", "standard", "comprehensive")
            
        Returns:
            EnhancedDocumentResult: Comprehensive document analysis
        """
        processing_start = time.time()
        document_id = f"doc_{int(time.time() * 1000)}"
        
        # Execute traditional processing in parallel with AI analysis
        traditional_task = asyncio.create_task(
            self._traditional_document_processing(document_path, document_type)
        )
        
        ai_analysis_task = asyncio.create_task(
            self._ai_powered_document_analysis(document_path, document_type, analysis_depth)
        )
        
        # Wait for both processing streams to complete
        traditional_result, ai_analysis = await asyncio.gather(
            traditional_task, ai_analysis_task
        )
        
        # Perform semantic search if vector stores are available
        semantic_results = await self._perform_semantic_search(
            traditional_result, ai_analysis
        )
        
        # Extract document intelligence
        document_intelligence = await self._extract_document_intelligence(
            traditional_result, ai_analysis, semantic_results
        )
        
        # Calculate processing quality
        processing_quality = await self._calculate_processing_quality(
            traditional_result, ai_analysis, document_intelligence
        )
        
        return EnhancedDocumentResult(
            document_id=document_id,
            traditional_processing=traditional_result,
            semantic_search_results=semantic_results,
            ai_insights=ai_analysis,
            document_intelligence=document_intelligence,
            processing_quality=processing_quality,
            processing_timestamp=time.time()
        )
    
    async def _traditional_document_processing(self, 
                                             document_path: Union[str, Path],
                                             document_type: str) -> Dict[str, Any]:
        """Execute traditional document processing."""
        
        try:
            # Use existing enhanced book processor
            processed_document = await self.traditional_processor.process_file(
                str(document_path)
            )
            
            return {
                "processing_method": "traditional_enhanced",
                "document_metadata": processed_document.get("metadata", {}),
                "extracted_text": processed_document.get("content", ""),
                "chunking_strategy": processed_document.get("chunking_info", {}),
                "traditional_quality_score": processed_document.get("quality_score", 0.8),
                "processing_status": "completed"
            }
            
        except Exception as e:
            return {
                "processing_method": "traditional_enhanced",
                "processing_status": "failed",
                "error": str(e),
                "traditional_quality_score": 0.0
            }
    
    async def _ai_powered_document_analysis(self, 
                                          document_path: Union[str, Path],
                                          document_type: str,
                                          analysis_depth: str) -> List[Dict[str, Any]]:
        """Perform AI-powered document analysis using OpenAI agents."""
        
        if not self.vector_store_ids:
            # If no vector stores configured, return simulated analysis
            return await self._simulated_ai_analysis(document_path, document_type)
        
        ai_insights = []
        
        # Generate analysis queries based on document type
        analysis_queries = self._generate_analysis_queries(document_type, analysis_depth)
        
        for query_info in analysis_queries:
            try:
                with trace(f"Document analysis: {query_info['query']}"):
                    # Build analysis prompt
                    analysis_prompt = self._build_analysis_prompt(query_info, document_type)
                    
                    # Execute analysis using OpenAI Agent
                    result = await Runner.run(
                        starting_agent=self.document_agent,
                        input=analysis_prompt
                    )
                    
                    # Process analysis result
                    processed_insight = await self._process_analysis_result(
                        result, query_info
                    )
                    
                    ai_insights.append(processed_insight)
                    
            except Exception as e:
                print(f"AI analysis failed for query {query_info['query']}: {e}")
                continue
        
        return ai_insights
    
    async def _simulated_ai_analysis(self, 
                                   document_path: Union[str, Path],
                                   document_type: str) -> List[Dict[str, Any]]:
        """Provide simulated AI analysis when vector stores are not available."""
        
        # Read document content for basic analysis
        try:
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            content = ""
        
        return [
            {
                "query_type": "content_summary",
                "analysis_result": f"Document analysis for {document_type} content",
                "key_insights": [
                    "Document processed successfully",
                    "Content extracted and analyzed",
                    "Ready for semantic search integration"
                ],
                "confidence_score": 0.7,
                "processing_method": "simulated",
                "content_length": len(content),
                "timestamp": time.time()
            },
            {
                "query_type": "concept_extraction",
                "analysis_result": "Key concepts identified from document",
                "key_insights": [
                    "Trading concepts present" if "trading" in content.lower() else "General financial content",
                    "Document contains structured information",
                    "Suitable for knowledge base integration"
                ],
                "confidence_score": 0.6,
                "processing_method": "simulated",
                "entities_found": content.count('.'),  # Simple metric
                "timestamp": time.time()
            }
        ]
    
    def _generate_analysis_queries(self, document_type: str, analysis_depth: str) -> List[Dict[str, Any]]:
        """Generate analysis queries based on document type and depth."""
        
        base_queries = [
            {
                "query": "document summary and key concepts",
                "query_type": "content_summary",
                "priority": "high",
                "expected_insights": ["main_themes", "key_concepts", "document_structure"]
            },
            {
                "query": "actionable insights and recommendations",
                "query_type": "actionable_extraction",
                "priority": "high",
                "expected_insights": ["recommendations", "action_items", "decision_points"]
            }
        ]
        
        # Add document-type specific queries
        strategy = self.analysis_strategies.get(document_type, self.analysis_strategies["financial_reports"])
        
        for focus_area in strategy["focus"]:
            base_queries.append({
                "query": f"{focus_area} analysis and insights",
                "query_type": f"{focus_area}_analysis",
                "priority": "medium",
                "expected_insights": strategy["extraction_targets"]
            })
        
        # Add depth-specific queries
        if analysis_depth == "comprehensive":
            base_queries.extend([
                {
                    "query": "risk assessment and potential issues",
                    "query_type": "risk_analysis",
                    "priority": "high",
                    "expected_insights": ["risk_factors", "mitigation_strategies", "impact_assessment"]
                },
                {
                    "query": "entity relationships and network analysis",
                    "query_type": "relationship_mapping",
                    "priority": "medium",
                    "expected_insights": ["entity_connections", "influence_networks", "dependency_analysis"]
                }
            ])
        
        return base_queries
    
    def _build_analysis_prompt(self, query_info: Dict[str, Any], document_type: str) -> str:
        """Build optimized analysis prompt for document processing."""
        
        query = query_info["query"]
        query_type = query_info["query_type"]
        expected_insights = query_info.get("expected_insights", [])
        
        prompt = f"""
        Analyze the documents for: "{query}"
        
        Document Type: {document_type}
        Analysis Focus: {query_type}
        
        Please provide comprehensive analysis including:
        
        1. Key Findings:
           - {expected_insights[0] if expected_insights else 'Primary insights'}
           - {expected_insights[1] if len(expected_insights) > 1 else 'Supporting details'}
           - {expected_insights[2] if len(expected_insights) > 2 else 'Additional context'}
        
        2. Structured Analysis:
           - Main concepts and themes
           - Quantitative data and metrics
           - Qualitative assessments
           - Risk factors and considerations
        
        3. Actionable Insights:
           - Specific recommendations
           - Implementation guidance
           - Priority levels
           - Success metrics
        
        4. Context and Relationships:
           - Connections to other concepts
           - Dependencies and prerequisites
           - Potential conflicts or synergies
        
        5. Confidence Assessment:
           - Reliability of findings
           - Data quality indicators
           - Uncertainty factors
        
        Focus on trading and financial applications. Provide specific, actionable insights
        with confidence scores and clear reasoning.
        """
        
        return prompt
    
    async def _process_analysis_result(self, 
                                     result: Any, 
                                     query_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure AI analysis results."""
        
        analysis_content = result.final_output
        
        # Extract key insights from the analysis
        key_insights = await self._extract_key_insights(analysis_content)
        
        # Calculate confidence score
        confidence_score = await self._calculate_analysis_confidence(analysis_content)
        
        # Extract entities and relationships
        entities = await self._extract_entities(analysis_content)
        
        # Assess sentiment and risk
        sentiment_risk = await self._assess_sentiment_and_risk(analysis_content)
        
        return {
            "query_type": query_info["query_type"],
            "query": query_info["query"],
            "analysis_result": analysis_content,
            "key_insights": key_insights,
            "confidence_score": confidence_score,
            "entities_extracted": entities,
            "sentiment_risk_assessment": sentiment_risk,
            "processing_method": "openai_file_search",
            "timestamp": time.time()
        }
    
    async def _extract_key_insights(self, analysis_content: str) -> List[str]:
        """Extract key insights from analysis content."""
        
        insights = []
        
        # Look for numbered lists or bullet points
        lines = analysis_content.split('\n')
        for line in lines:
            line = line.strip()
            if (line.startswith('- ') or 
                line.startswith('• ') or 
                any(line.startswith(f'{i}.') for i in range(1, 10))):
                insight = line.lstrip('- •123456789. ').strip()
                if len(insight) > 10:  # Filter out very short insights
                    insights.append(insight)
        
        # If no structured insights found, extract sentences with keywords
        if not insights:
            sentences = analysis_content.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in 
                      ['recommend', 'insight', 'important', 'key', 'critical', 'should']):
                    insights.append(sentence.strip())
        
        return insights[:10]  # Limit to top 10 insights
    
    async def _calculate_analysis_confidence(self, analysis_content: str) -> float:
        """Calculate confidence score for AI analysis."""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence for detailed analysis
        if len(analysis_content) > 500:
            confidence += 0.2
        
        # Increase confidence for structured content
        if any(marker in analysis_content for marker in ['1.', '2.', '3.', '-', '•']):
            confidence += 0.15
        
        # Increase confidence for quantitative data
        import re
        if re.search(r'\d+%|\$\d+|\d+\.\d+', analysis_content):
            confidence += 0.1
        
        # Increase confidence for specific recommendations
        if any(word in analysis_content.lower() for word in 
              ['recommend', 'suggest', 'should', 'propose']):
            confidence += 0.1
        
        # Decrease confidence for uncertainty markers
        if any(word in analysis_content.lower() for word in 
              ['uncertain', 'unclear', 'maybe', 'might', 'possibly']):
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    async def _extract_entities(self, analysis_content: str) -> List[Dict[str, Any]]:
        """Extract entities from analysis content."""
        
        entities = []
        content_lower = analysis_content.lower()
        
        # Financial entities
        financial_terms = ['stock', 'bond', 'option', 'future', 'etf', 'portfolio', 'risk', 'return']
        for term in financial_terms:
            if term in content_lower:
                entities.append({
                    "entity": term,
                    "type": "financial_instrument",
                    "confidence": 0.8
                })
        
        # Company/organization entities (simple heuristic)
        import re
        company_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'
        matches = re.findall(company_pattern, analysis_content)
        for match in matches:
            if len(match.split()) <= 3:  # Reasonable company name length
                entities.append({
                    "entity": match,
                    "type": "organization",
                    "confidence": 0.6
                })
        
        return entities[:20]  # Limit to top 20 entities
    
    async def _assess_sentiment_and_risk(self, analysis_content: str) -> Dict[str, Any]:
        """Assess sentiment and risk from analysis content."""
        
        content_lower = analysis_content.lower()
        
        # Sentiment analysis
        positive_words = ['positive', 'good', 'excellent', 'strong', 'growth', 'opportunity']
        negative_words = ['negative', 'poor', 'weak', 'decline', 'risk', 'threat', 'loss']
        
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count + negative_count > 0:
            sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
        else:
            sentiment_score = 0.0
        
        sentiment_label = "positive" if sentiment_score > 0.2 else "negative" if sentiment_score < -0.2 else "neutral"
        
        # Risk assessment
        risk_indicators = ['risk', 'volatility', 'uncertainty', 'threat', 'challenge', 'concern']
        risk_count = sum(1 for indicator in risk_indicators if indicator in content_lower)
        risk_level = "high" if risk_count > 3 else "medium" if risk_count > 1 else "low"
        
        return {
            "sentiment": {
                "label": sentiment_label,
                "score": sentiment_score,
                "confidence": 0.7
            },
            "risk_assessment": {
                "level": risk_level,
                "indicators_found": risk_count,
                "confidence": 0.75
            }
        }
    
    async def _perform_semantic_search(self, 
                                     traditional_result: Dict[str, Any],
                                     ai_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform semantic search using extracted insights."""
        
        if not self.vector_store_ids:
            return []
        
        semantic_results = []
        
        # Extract search queries from AI analysis
        search_queries = []
        for insight in ai_analysis:
            for key_insight in insight.get("key_insights", [])[:3]:  # Top 3 insights per analysis
                if len(key_insight) > 20:  # Meaningful insights only
                    search_queries.append(key_insight)
        
        # Perform semantic searches
        for query in search_queries[:5]:  # Limit to 5 searches
            try:
                search_prompt = f"""
                Search for documents related to: "{query}"
                
                Find relevant documents that contain information about this topic.
                Provide a summary of the key findings and how they relate to the query.
                """
                
                with trace(f"Semantic search: {query[:50]}..."):
                    result = await Runner.run(
                        starting_agent=self.document_agent,
                        input=search_prompt
                    )
                    
                    semantic_results.append({
                        "query": query,
                        "search_results": result.final_output,
                        "relevance_score": 0.8,  # Would calculate based on actual results
                        "timestamp": time.time()
                    })
                    
            except Exception as e:
                print(f"Semantic search failed for query: {query[:50]}... Error: {e}")
                continue
        
        return semantic_results
    
    async def _extract_document_intelligence(self, 
                                           traditional_result: Dict[str, Any],
                                           ai_analysis: List[Dict[str, Any]],
                                           semantic_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract comprehensive document intelligence."""
        
        # Aggregate key concepts from all sources
        key_concepts = []
        for insight in ai_analysis:
            key_concepts.extend(insight.get("key_insights", []))
        
        # Aggregate entities
        all_entities = []
        for insight in ai_analysis:
            all_entities.extend(insight.get("entities_extracted", []))
        
        # Calculate overall sentiment
        sentiment_scores = []
        for insight in ai_analysis:
            sentiment_data = insight.get("sentiment_risk_assessment", {}).get("sentiment", {})
            if "score" in sentiment_data:
                sentiment_scores.append(sentiment_data["score"])
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        
        # Assess complexity
        total_content_length = sum(len(insight.get("analysis_result", "")) for insight in ai_analysis)
        complexity_score = min(total_content_length / 5000, 1.0)  # Normalize to 0-1
        
        # Calculate overall confidence
        confidence_scores = [insight.get("confidence_score", 0.5) for insight in ai_analysis]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        return {
            "key_concepts": list(set(key_concepts))[:20],  # Top 20 unique concepts
            "sentiment_analysis": {
                "overall_sentiment": "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral",
                "sentiment_score": avg_sentiment,
                "confidence": 0.8
            },
            "complexity_assessment": {
                "complexity_score": complexity_score,
                "complexity_level": "high" if complexity_score > 0.7 else "medium" if complexity_score > 0.3 else "low",
                "content_volume": total_content_length
            },
            "topic_classification": await self._classify_topics(key_concepts),
            "entity_extraction": all_entities[:30],  # Top 30 entities
            "relationship_mapping": await self._map_relationships(all_entities),
            "actionable_insights": await self._extract_actionable_insights(ai_analysis),
            "confidence_score": avg_confidence
        }
    
    async def _classify_topics(self, key_concepts: List[str]) -> List[str]:
        """Classify document topics based on key concepts."""
        
        topics = set()
        
        for concept in key_concepts:
            concept_lower = concept.lower()
            
            if any(term in concept_lower for term in ['trading', 'strategy', 'signal']):
                topics.add("trading_strategies")
            elif any(term in concept_lower for term in ['risk', 'volatility', 'drawdown']):
                topics.add("risk_management")
            elif any(term in concept_lower for term in ['market', 'trend', 'analysis']):
                topics.add("market_analysis")
            elif any(term in concept_lower for term in ['regulation', 'compliance', 'legal']):
                topics.add("regulatory_compliance")
            elif any(term in concept_lower for term in ['technology', 'algorithm', 'system']):
                topics.add("financial_technology")
            else:
                topics.add("general_finance")
        
        return list(topics)
    
    async def _map_relationships(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Map relationships between extracted entities."""
        
        relationships = {
            "entity_count": len(entities),
            "entity_types": {},
            "connection_strength": "medium",  # Would calculate based on co-occurrence
            "network_density": 0.6  # Simulated network density
        }
        
        # Count entity types
        for entity in entities:
            entity_type = entity.get("type", "unknown")
            relationships["entity_types"][entity_type] = relationships["entity_types"].get(entity_type, 0) + 1
        
        return relationships
    
    async def _extract_actionable_insights(self, ai_analysis: List[Dict[str, Any]]) -> List[str]:
        """Extract actionable insights from AI analysis."""
        
        actionable_insights = []
        
        for insight in ai_analysis:
            for key_insight in insight.get("key_insights", []):
                if any(action_word in key_insight.lower() for action_word in 
                      ['should', 'recommend', 'consider', 'implement', 'adopt']):
                    actionable_insights.append(key_insight)
        
        return list(set(actionable_insights))[:15]  # Top 15 unique actionable insights
    
    async def _calculate_processing_quality(self, 
                                          traditional_result: Dict[str, Any],
                                          ai_analysis: List[Dict[str, Any]],
                                          document_intelligence: Dict[str, Any]) -> float:
        """Calculate overall processing quality score."""
        
        # Factor 1: Traditional processing quality
        traditional_quality = traditional_result.get("traditional_quality_score", 0.5)
        
        # Factor 2: AI analysis quality
        ai_quality_scores = [insight.get("confidence_score", 0.5) for insight in ai_analysis]
        avg_ai_quality = sum(ai_quality_scores) / len(ai_quality_scores) if ai_quality_scores else 0.5
        
        # Factor 3: Document intelligence confidence
        intelligence_confidence = document_intelligence.get("confidence_score", 0.5)
        
        # Factor 4: Analysis comprehensiveness
        comprehensiveness = min(len(ai_analysis) / 5, 1.0)  # Normalize based on number of analyses
        
        # Weighted quality score
        quality_score = (
            traditional_quality * 0.3 +
            avg_ai_quality * 0.3 +
            intelligence_confidence * 0.25 +
            comprehensiveness * 0.15
        )
        
        return quality_score
    
    def get_enhanced_capabilities(self) -> List[str]:
        """Return enhanced document processing capabilities."""
        return self.enhanced_capabilities