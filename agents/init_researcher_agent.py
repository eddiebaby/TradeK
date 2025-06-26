#!/usr/bin/env python3
"""
Initialize and Invoke the Researcher Agent for Schwab OAuth Research

This script initializes the Researcher Agent to research the Schwab OAuth authentication protocol
and analyze the documentation at https://developer.schwab.com/user-guides/get-started/authenticate-with-oauth
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any

# Add agents directory to Python path
agents_dir = Path(__file__).parent / "agents"
sys.path.append(str(agents_dir))

try:
    from researcher.researcher_agent import ResearcherAgent, ResearchDomain
    from core.agent_base import TaskContext, AgentRole
    print("✅ Successfully imported standard Researcher Agent")
    
    # Try to import enhanced researcher with web capabilities
    try:
        from researcher.enhanced_researcher_agent import EnhancedResearcherAgent
        print("✅ Enhanced Researcher Agent with web search available")
        USE_ENHANCED = True
    except ImportError as e:
        print(f"⚠️  Enhanced Researcher Agent not available: {e}")
        print("📝 Falling back to standard Researcher Agent")
        USE_ENHANCED = False
        
except ImportError as e:
    print(f"❌ Failed to import Researcher Agent: {e}")
    print("💡 Make sure you're running from the TradeKnowledge root directory")
    sys.exit(1)


class SchwabOAuthResearcher:
    """Specialized researcher for Schwab OAuth authentication analysis."""
    
    def __init__(self):
        """Initialize the researcher agent."""
        if USE_ENHANCED:
            # Use enhanced agent with web search capabilities
            self.agent = EnhancedResearcherAgent()
            print("🚀 Initialized Enhanced Researcher Agent with web search capabilities")
        else:
            # Use standard researcher agent
            self.agent = ResearcherAgent()
            print("🚀 Initialized Standard Researcher Agent")
        
        self.research_results = {}
    
    async def research_schwab_oauth(self) -> Dict[str, Any]:
        """
        Research Schwab OAuth authentication protocol.
        
        Returns:
            Dict containing comprehensive research results
        """
        print("\n🔍 STARTING SCHWAB OAUTH RESEARCH")
        print("=" * 50)
        
        # Define research specification
        research_spec = {
            "domains": ["technical_deep_dive", "security_intelligence", "best_practices"],
            "focus_areas": [
                "OAuth 2.0 implementation",
                "Schwab API authentication",
                "Financial services security",
                "API integration patterns",
                "Token management",
                "Security best practices"
            ],
            "depth": "comprehensive",
            "context": {
                "target_platform": "Schwab Developer API",
                "authentication_method": "OAuth 2.0",
                "integration_type": "Trading application",
                "security_requirements": "Financial services compliance"
            },
            "priority": 1
        }
        
        # Add web-specific research if enhanced agent is available
        if USE_ENHANCED:
            research_spec.update({
                "web_sources": [
                    "https://developer.schwab.com/user-guides/get-started/authenticate-with-oauth",
                    "https://developer.schwab.com/products/trader-api--individual/details/documentation/Trader%20API%20Guide.json",
                    "https://developer.schwab.com",
                    "https://oauth.net/2/",
                    "https://tools.ietf.org/html/rfc6749"
                ],
                "github_query": "OAuth 2.0 financial API Python implementation",
                "content_type": "technical_docs",
                "insight_type": "implementation"
            })
        
        # Create task context
        task_context = TaskContext(
            task_id="schwab_oauth_research",
            task_type="research",
            requirements=research_spec,
            quality_requirements={
                "research_accuracy": 95,
                "source_diversity": 5,
                "confidence_threshold": 0.85
            },
            context=research_spec["context"]
        )
        
        # Execute research
        if USE_ENHANCED:
            print("🌐 Executing enhanced research with web intelligence...")
            results = await self.agent.conduct_enhanced_research(research_spec)
            self.research_results = self._format_enhanced_results(results)
        else:
            print("📚 Executing standard comprehensive research...")
            results = await self.agent.process_task(task_context)
            self.research_results = results
        
        # Generate specific OAuth implementation guidance
        oauth_guidance = await self._generate_oauth_implementation_guidance()
        self.research_results["oauth_implementation_guidance"] = oauth_guidance
        
        return self.research_results
    
    def _format_enhanced_results(self, enhanced_results) -> Dict[str, Any]:
        """Format enhanced research results for easier consumption."""
        
        return {
            "research_type": "enhanced_web_enabled",
            "traditional_research": enhanced_results.traditional_research.__dict__,
            "web_intelligence": enhanced_results.web_intelligence,
            "market_trends": enhanced_results.market_trends,
            "real_time_insights": enhanced_results.real_time_insights,
            "synthesis_quality": enhanced_results.synthesis_quality,
            "research_timestamp": enhanced_results.research_timestamp
        }
    
    async def _generate_oauth_implementation_guidance(self) -> Dict[str, Any]:
        """Generate specific OAuth implementation guidance for Schwab API."""
        
        return {
            "implementation_steps": [
                "1. Register application with Schwab Developer Portal",
                "2. Configure OAuth 2.0 redirect URIs",
                "3. Implement authorization code flow with PKCE",
                "4. Handle token refresh and expiration",
                "5. Implement proper error handling and retry logic",
                "6. Add comprehensive logging and monitoring"
            ],
            "security_considerations": [
                "Use HTTPS for all OAuth flows",
                "Implement PKCE (Proof Key for Code Exchange)",
                "Store tokens securely (encrypted at rest)",
                "Implement token rotation and refresh",
                "Add rate limiting and circuit breakers",
                "Monitor for suspicious authentication patterns"
            ],
            "code_structure": {
                "suggested_files": [
                    "schwab_oauth_client.py",
                    "token_manager.py",
                    "auth_middleware.py",
                    "schwab_api_client.py"
                ],
                "key_classes": [
                    "SchwabOAuthClient",
                    "TokenManager", 
                    "AuthenticationMiddleware",
                    "SchwabAPIClient"
                ]
            },
            "testing_strategy": [
                "Unit tests for OAuth flow components",
                "Integration tests with Schwab sandbox",
                "Security tests for token handling",
                "Performance tests for API calls",
                "Error handling and retry logic tests"
            ]
        }
    
    def display_results(self):
        """Display research results in a readable format."""
        
        print("\n📊 SCHWAB OAUTH RESEARCH RESULTS")
        print("=" * 50)
        
        if not self.research_results:
            print("❌ No research results available")
            return
        
        # Display key insights
        if "actionable_insights" in self.research_results:
            insights = self.research_results["actionable_insights"]
            print(f"\n💡 KEY INSIGHTS ({len(insights)} found):")
            for i, insight in enumerate(insights[:5], 1):
                print(f"  {i}. {insight.get('title', 'Insight')}")
                if 'actions' in insight:
                    for action in insight['actions'][:2]:
                        print(f"     • {action}")
        
        # Display OAuth implementation guidance
        if "oauth_implementation_guidance" in self.research_results:
            guidance = self.research_results["oauth_implementation_guidance"]
            print(f"\n🔐 OAUTH IMPLEMENTATION GUIDANCE:")
            print(f"  📋 Implementation Steps: {len(guidance['implementation_steps'])}")
            print(f"  🔒 Security Considerations: {len(guidance['security_considerations'])}")
            print(f"  🧪 Testing Strategy: {len(guidance['testing_strategy'])} areas")
        
        # Display research quality metrics
        if USE_ENHANCED:
            print(f"\n📈 RESEARCH QUALITY:")
            print(f"  🌐 Web Intelligence: Available")
            print(f"  🔗 Synthesis Quality: {self.research_results.get('synthesis_quality', 0):.2f}")
            if 'real_time_insights' in self.research_results:
                print(f"  ⏱️  Real-time Insights: {len(self.research_results['real_time_insights'])}")
        else:
            if "research_metrics" in self.research_results:
                metrics = self.research_results["research_metrics"]
                print(f"\n📈 RESEARCH QUALITY:")
                print(f"  ⏱️  Duration: {metrics.get('research_duration', 0):.2f}s")
                print(f"  💡 Insights: {metrics.get('insight_count', 0)}")
                print(f"  🎯 Confidence: {metrics.get('confidence_average', 0):.2f}")
    
    def save_results(self, output_file: str = "schwab_oauth_research.json"):
        """Save research results to file."""
        
        output_path = Path(output_file)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.research_results, f, indent=2, default=str)
            print(f"\n💾 Research results saved to: {output_path.absolute()}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def get_oauth_implementation_plan(self) -> Dict[str, Any]:
        """Get a focused OAuth implementation plan."""
        
        if not self.research_results:
            return {"error": "No research results available"}
        
        guidance = self.research_results.get("oauth_implementation_guidance", {})
        
        return {
            "phase_1_setup": {
                "tasks": guidance.get("implementation_steps", [])[:3],
                "duration": "1-2 days",
                "priority": "high"
            },
            "phase_2_implementation": {
                "tasks": guidance.get("implementation_steps", [])[3:],
                "duration": "3-5 days", 
                "priority": "high"
            },
            "security_checklist": guidance.get("security_considerations", []),
            "testing_plan": guidance.get("testing_strategy", []),
            "recommended_architecture": guidance.get("code_structure", {})
        }


async def main():
    """Main entry point for Schwab OAuth research."""
    
    print("🤖 SCHWAB OAUTH RESEARCHER INITIALIZATION")
    print("=" * 50)
    
    # Initialize researcher
    researcher = SchwabOAuthResearcher()
    
    try:
        # Execute research
        results = await researcher.research_schwab_oauth()
        
        # Display results
        researcher.display_results()
        
        # Save results
        researcher.save_results()
        
        # Get implementation plan
        impl_plan = researcher.get_oauth_implementation_plan()
        
        print("\n🚀 IMPLEMENTATION PLAN READY")
        print("=" * 30)
        print("Next steps:")
        print("1. Review the saved research results")
        print("2. Follow the OAuth implementation guidance")
        print("3. Use the provided security checklist")
        print("4. Implement the recommended architecture")
        
        return {
            "status": "success",
            "research_results": results,
            "implementation_plan": impl_plan
        }
        
    except Exception as e:
        print(f"❌ Research failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    """Run the Schwab OAuth researcher."""
    
    # Check if we're in the right directory
    if not Path("agents").exists():
        print("❌ Error: 'agents' directory not found")
        print("💡 Please run this script from the TradeKnowledge root directory")
        sys.exit(1)
    
    # Run the research
    try:
        results = asyncio.run(main())
        
        if results["status"] == "success":
            print("\n✅ Schwab OAuth research completed successfully!")
            print("📁 Check 'schwab_oauth_research.json' for detailed results")
        else:
            print(f"\n❌ Research failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Research interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)