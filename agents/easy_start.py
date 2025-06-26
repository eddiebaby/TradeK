#!/usr/bin/env python3
"""
Easy Start Guide - How to Use MASTERMIND & EXECUTOR Agents
Run this script to see the agents in action!
"""

import asyncio
import sys
import os
from pathlib import Path

# Add agents directory to Python path
sys.path.append(str(Path(__file__).parent))

from agent_orchestrator import AgentOrchestrator

async def quick_example():
    """Quick 5-minute example - Build a REST API endpoint."""
    
    print("🚀 QUICK START: Building a REST API Endpoint")
    print("=" * 50)
    
    # Initialize the agent system
    orchestrator = AgentOrchestrator()
    
    # Define what you want to build
    requirement = """
    Build a REST API endpoint for user authentication that:
    - Accepts email/password login
    - Returns JWT tokens
    - Includes rate limiting
    - Has comprehensive security
    """
    
    project_context = {
        "project_type": "rest_api",
        "technology_stack": "FastAPI + SQLAlchemy + PostgreSQL",
        "deployment_target": "Docker + Kubernetes",
        "security_level": "high"
    }
    
    quality_requirements = {
        "test_coverage": 95,
        "mutation_score": 85,
        "performance": {"max_response_time": 200},
        "security": {"min_security_score": 9.0}
    }
    
    print(f"📋 REQUIREMENT: {requirement.strip()}")
    print(f"🛠️  TECH STACK: {project_context['technology_stack']}")
    print(f"🎯 QUALITY TARGETS: {quality_requirements}")
    print("\n" + "=" * 50)
    
    # Execute the development cycle
    print("⚡ AGENTS STARTING WORK...")
    
    results = await orchestrator.execute_comprehensive_development_cycle(
        requirement=requirement,
        project_context=project_context,
        quality_requirements=quality_requirements
    )
    
    # Show results
    print("\n🎉 RESULTS:")
    print("=" * 50)
    print(f"✅ Quality Score: {results['session_results']['metrics'].quality_amplification:.2f}/10")
    print(f"🤝 Collaboration Effectiveness: {results['session_results']['metrics'].collaboration_effectiveness:.2f}")
    print(f"📊 Test Coverage: {results['session_results']['phase_results']['execution']['implementation_metrics']['test_completeness']:.1f}%")
    print(f"🔒 Security Score: {results['session_results']['phase_results']['execution']['implementation_metrics']['security_compliance']:.1f}/10")
    
    return results

async def custom_example():
    """Custom example - You define what to build."""
    
    print("\n" + "🎯 CUSTOM PROJECT EXAMPLE")
    print("=" * 50)
    
    # Get user input
    print("What do you want the agents to build?")
    print("Examples:")
    print("- 'Build a file upload service with virus scanning'")
    print("- 'Create a real-time chat system with WebSockets'") 
    print("- 'Implement a recommendation engine with ML'")
    print("- 'Build a microservice for payment processing'")
    
    requirement = input("\n📝 Enter your requirement: ").strip()
    
    if not requirement:
        requirement = "Build a file upload service with virus scanning and storage management"
        print(f"Using default: {requirement}")
    
    # Auto-configure based on requirement
    if "payment" in requirement.lower():
        tech_stack = "FastAPI + Stripe + PostgreSQL"
        security_level = "maximum"
    elif "chat" in requirement.lower() or "websocket" in requirement.lower():
        tech_stack = "FastAPI + WebSockets + Redis"
        security_level = "high"
    elif "ml" in requirement.lower() or "recommendation" in requirement.lower():
        tech_stack = "FastAPI + TensorFlow + Vector DB"
        security_level = "medium"
    else:
        tech_stack = "FastAPI + PostgreSQL + Redis"
        security_level = "high"
    
    orchestrator = AgentOrchestrator()
    
    project_context = {
        "project_type": "microservice",
        "technology_stack": tech_stack,
        "deployment_target": "Kubernetes",
        "security_level": security_level
    }
    
    quality_requirements = {
        "test_coverage": 90,
        "mutation_score": 80,
        "performance": {"max_response_time": 150},
        "security": {"min_security_score": 8.5}
    }
    
    print(f"\n🤖 AGENTS WILL BUILD: {requirement}")
    print(f"🛠️  USING TECH STACK: {tech_stack}")
    print("\n⚡ Starting development cycle...")
    
    results = await orchestrator.execute_comprehensive_development_cycle(
        requirement=requirement,
        project_context=project_context,
        quality_requirements=quality_requirements
    )
    
    print(f"\n✅ COMPLETED! Quality Score: {results['session_results']['metrics'].quality_amplification:.2f}/10")
    
    return results

def interactive_menu():
    """Interactive menu for choosing how to use the agents."""
    
    print("🤖 MASTERMIND & EXECUTOR Agent System")
    print("=" * 40)
    print("Choose how you want to use the agents:")
    print()
    print("1. 🚀 Quick Demo (5 minutes) - REST API example")
    print("2. 🎯 Custom Project - Tell agents what to build")
    print("3. 📚 View Agent Capabilities")
    print("4. 🔧 Advanced Collaboration Workflow")
    print("5. 📖 Help & Documentation")
    print()
    
    choice = input("Enter choice (1-5): ").strip()
    return choice

async def show_capabilities():
    """Show what the agents can do."""
    
    print("\n🧠 MASTERMIND AGENT CAPABILITIES:")
    print("=" * 40)
    print("Strategic Analysis:")
    print("  • Architectural design & pattern recognition")
    print("  • Technical debt assessment & prioritization") 
    print("  • Performance bottleneck prediction")
    print("  • Security threat modeling")
    print("  • Technology stack optimization")
    print()
    print("Quality Orchestration:")
    print("  • Custom TDD workflow design")
    print("  • Test strategy development")
    print("  • Quality gate definition")
    print("  • Risk assessment & mitigation")
    
    print("\n⚡ EXECUTOR AGENT CAPABILITIES:")
    print("=" * 40)
    print("Implementation Excellence:")
    print("  • Test-driven development (TDD)")
    print("  • High-quality code generation")
    print("  • Performance optimization")
    print("  • Refactoring & code improvement")
    print()
    print("Testing & DevOps:")
    print("  • 6 types of testing (unit, integration, mutation, property, chaos, security)")
    print("  • CI/CD pipeline generation")
    print("  • Container orchestration")
    print("  • Monitoring & alerting setup")
    
    print("\n🤝 COLLABORATION FEATURES:")
    print("=" * 40)
    print("  • Strategic Implementation Cycle (4 phases)")
    print("  • Continuous Quality Amplification")
    print("  • Intelligence learning & adaptation")
    print("  • Context preservation & handoffs")
    print("  • Real-time collaboration metrics")

async def advanced_workflow():
    """Show advanced workflow capabilities."""
    
    print("\n🔧 ADVANCED COLLABORATION WORKFLOW")
    print("=" * 50)
    
    orchestrator = AgentOrchestrator()
    
    # Example: Improvement cycle on existing code
    print("Example: Improving existing codebase")
    
    improvement_targets = {
        "focus_areas": ["performance", "security", "maintainability"],
        "target_improvements": {
            "performance": 30,  # 30% improvement
            "security_score": 9.5,
            "test_coverage": 95
        }
    }
    
    print("🔍 Analyzing current codebase...")
    
    # This would analyze your actual codebase
    codebase_path = "/home/scottschweizer/TradeKnowledge/src"
    
    improvement_results = await orchestrator.continuous_improvement_cycle(
        codebase_path=codebase_path,
        improvement_targets=improvement_targets
    )
    
    print(f"✅ Improvement analysis complete!")
    print(f"📊 Improvement opportunities found: {len(improvement_results['improvement_opportunities'])}")
    print(f"🎯 Prioritized improvements: {len(improvement_results['implemented_improvements'])}")

def show_help():
    """Show help documentation."""
    
    print("\n📖 HELP & DOCUMENTATION")
    print("=" * 40)
    print()
    print("GETTING STARTED:")
    print("1. Run this script: python easy_start.py")
    print("2. Choose option 1 for quick demo")
    print("3. Try option 2 to build your own project")
    print()
    print("WHAT THE AGENTS DO:")
    print("• MASTERMIND: Strategic thinking, architecture, quality planning")
    print("• EXECUTOR: Implementation, testing, DevOps, deployment")
    print("• TOGETHER: Create better code than either could alone")
    print()
    print("TYPICAL WORKFLOW:")
    print("1. You provide a requirement (what to build)")
    print("2. MASTERMIND analyzes and designs architecture")
    print("3. EXECUTOR implements with TDD and comprehensive testing")
    print("4. MASTERMIND reviews and provides feedback")
    print("5. System delivers production-ready code with 90%+ quality")
    print()
    print("INTEGRATION:")
    print("• Can integrate with existing projects")
    print("• Works with any Python codebase")
    print("• Generates deployable code and infrastructure")
    print("• Provides CI/CD pipelines and monitoring")

async def main():
    """Main entry point."""
    
    while True:
        choice = interactive_menu()
        
        if choice == "1":
            await quick_example()
        elif choice == "2":
            await custom_example()
        elif choice == "3":
            await show_capabilities()
        elif choice == "4":
            await advanced_workflow()
        elif choice == "5":
            show_help()
        else:
            print("Invalid choice. Please enter 1-5.")
            continue
        
        print("\n" + "=" * 50)
        continue_choice = input("Want to try something else? (y/n): ").lower()
        if continue_choice != 'y':
            break
    
    print("\n👋 Thanks for using MASTERMIND & EXECUTOR!")
    print("💡 Tip: Run 'python easy_start.py' anytime to use the agents")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure you're in the agents directory and all files are present")