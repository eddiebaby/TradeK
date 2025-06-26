#!/usr/bin/env python3
"""
Interactive script to give research tasks to RESEARCHER agent
Now with InfluxDB blackboard integration for trio collaboration
"""
import asyncio
import sys
import json
import time
from pathlib import Path

# Add agents directory to path
sys.path.append(str(Path(__file__).parent))

from researcher.researcher_agent import ResearcherAgent
from core.agent_base import TaskContext
from influx_blackboard import write_task, read_tasks, update_status, log_performance, write_reflection, get_context

async def main():
    print("🔍 RESEARCHER Intelligence Agent")
    print("=" * 50)
    
    researcher = ResearcherAgent()
    
    print("🧠 Research Capabilities:")
    capabilities = researcher.get_capabilities()
    for i, capability in enumerate(capabilities[:6], 1):
        print(f"   {i}. {capability.replace('_', ' ').title()}")
    
    print("\n🔬 Research Modes:")
    modes = researcher.get_research_modes()
    for mode, description in list(modes.items())[:4]:
        print(f"   • {mode}: {description}")
    
    while True:
        print("\n" + "="*60)
        print("What would you like RESEARCHER to investigate?")
        print("(Type 'quit' to exit, 'help' for examples)")
        
        task = input("🔍 Enter research topic: ").strip()
        
        if task.lower() in ['quit', 'exit', 'q']:
            break
            
        if task.lower() == 'help':
            print("\n📝 Example Research Topics:")
            print("   • 'security best practices for API authentication'")
            print("   • 'performance optimization techniques for FastAPI'")
            print("   • 'microservices architecture patterns for trading systems'")
            print("   • 'testing strategies for real-time data processing'")
            print("   • 'emerging trends in AI-powered development'")
            continue
            
        if not task:
            continue
            
        print(f"\n🔍 RESEARCHER investigating: '{task}'")
        print("⏳ Gathering intelligence across multiple domains...")
        
        try:
            start_time = time.time()
            
            # Log research task start to blackboard
            await log_performance("RESEARCHER", "research_start", 50, 0.1, True)
            
            # Create research specification
            research_spec = {
                "domains": ["technical_analysis", "industry_standards", "vulnerability_research"],
                "focus_areas": [task],
                "depth": "comprehensive",
                "context": {"user_query": task},
                "priority": 1,
                "target_format": "general"
            }
            
            # Write research task to blackboard for trio coordination
            task_id = await write_task("RESEARCHER", "user_research_request", {
                "query": task,
                "spec": research_spec,
                "requester": "user",
                "timestamp": time.time()
            }, priority=1)
            
            print(f"📋 Task logged to blackboard: {task_id}")
            
            # Conduct research
            result = await researcher.targeted_research(research_spec)
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            tokens_used = len(str(result)) // 4  # Rough token estimate
            
            # Log performance to blackboard
            await log_performance("RESEARCHER", "targeted_research", tokens_used, execution_time, True, len(str(result)))
            
            # Update task status in blackboard
            await update_status(task_id, "completed", "RESEARCHER")
            
            print("\n✅ RESEARCHER Intelligence Complete!")
            print("-" * 50)
            
            # Extract and display key information
            research_intel = result.get("research_intelligence", {})
            
            if "summary" in research_intel:
                summary = research_intel["summary"]
                print(f"📊 Research Summary:")
                print(f"   • Total Insights: {summary.get('total_insights', 0)}")
                print(f"   • Average Confidence: {summary.get('average_confidence', 0):.2f}")
                print(f"   • Domains Covered: {', '.join(summary.get('domains_covered', []))}")
            
            if "insights" in research_intel:
                insights = research_intel["insights"][:3]  # Show top 3
                print(f"\n🔑 Key Insights:")
                for i, insight in enumerate(insights, 1):
                    print(f"   {i}. {insight.get('title', 'N/A')}")
                    print(f"      Confidence: {insight.get('confidence_score', 0):.2f}")
                    recs = insight.get('actionable_recommendations', [])
                    if recs:
                        print(f"      Top Recommendation: {recs[0]}")
            
            if "best_practices" in research_intel:
                practices = research_intel["best_practices"][:2]
                print(f"\n📋 Best Practices Found:")
                for practice in practices:
                    print(f"   • {practice.get('title', 'N/A')}")
                    print(f"     {practice.get('description', 'N/A')}")
            
            if "implementation_patterns" in research_intel:
                patterns = research_intel["implementation_patterns"][:2]
                print(f"\n🏗️  Implementation Patterns:")
                for pattern in patterns:
                    print(f"   • {pattern.get('pattern', 'N/A')}")
                    print(f"     Confidence: {pattern.get('confidence', 0):.2f}")
            
            # Show actionable insights
            actionable = result.get("actionable_insights", [])
            if actionable:
                print(f"\n⚡ Actionable Insights:")
                for insight in actionable[:3]:
                    print(f"   🎯 {insight.get('title', 'N/A')}")
                    print(f"      Priority: {insight.get('priority', 'N/A')}")
                    actions = insight.get('actions', [])
                    if actions:
                        print(f"      Action: {actions[0]}")
            
            print("-" * 50)
            
            # Write reflection on research quality to blackboard
            quality_score = research_intel.get("summary", {}).get("average_confidence", 0.5)
            await write_reflection(
                "RESEARCHER", 
                "performance", 
                "medium",
                f"Completed research for '{task}' with {quality_score:.2f} confidence",
                "Continue monitoring research quality trends",
                quality_score
            )
            
            # Check if we should hand off to MASTERMIND for strategic analysis
            if quality_score > 0.8 and actionable:
                print("🤝 High-quality research completed - preparing handoff to MASTERMIND...")
                handoff_task_id = await write_task("MASTERMIND", "strategic_analysis", {
                    "research_results": result,
                    "source_task": task_id,
                    "research_topic": task,
                    "confidence_level": quality_score,
                    "handoff_reason": "high_quality_research_for_strategy"
                }, priority=1)
                print(f"📤 MASTERMIND task created: {handoff_task_id}")
            
        except Exception as e:
            print(f"❌ Research Error: {e}")
            # Log error to blackboard
            await log_performance("RESEARCHER", "research_error", 0, 0, False)
            await write_reflection(
                "RESEARCHER",
                "error", 
                "high",
                f"Research failed for '{task}': {str(e)}",
                "Review error patterns and improve error handling",
                0.2
            )
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Research session ended.")
    except Exception as e:
        print(f"❌ Error: {e}")