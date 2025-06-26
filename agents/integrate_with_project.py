#!/usr/bin/env python3
"""
Integrate Agents with Your Existing Project
Add features, improve code quality, get strategic advice
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from agent_orchestrator import AgentOrchestrator

class ProjectAssistant:
    """Helper for integrating agents with existing projects."""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.orchestrator = AgentOrchestrator()
    
    async def add_feature(self, feature_description: str):
        """Add a new feature to your existing project."""
        
        print(f"🆕 Adding feature: {feature_description}")
        
        # Detect your project's tech stack (simplified)
        tech_stack = self._detect_tech_stack()
        
        project_context = {
            "project_type": "existing_project",
            "technology_stack": tech_stack,
            "project_path": self.project_path,
            "integration_mode": True
        }
        
        quality_requirements = {
            "test_coverage": 90,
            "mutation_score": 80,
            "integration_testing": True,
            "backward_compatibility": True
        }
        
        return await self.orchestrator.execute_comprehensive_development_cycle(
            requirement=f"Add feature to existing project: {feature_description}",
            project_context=project_context,
            quality_requirements=quality_requirements
        )
    
    async def improve_code_quality(self, focus_areas: list = None):
        """Improve existing code quality."""
        
        if not focus_areas:
            focus_areas = ["performance", "security", "maintainability", "testing"]
        
        print(f"🔧 Improving code quality in: {', '.join(focus_areas)}")
        
        improvement_targets = {
            "focus_areas": focus_areas,
            "target_improvements": {
                "performance": 25,  # 25% improvement
                "security_score": 9.0,
                "test_coverage": 90,
                "maintainability": 8.5
            }
        }
        
        return await self.orchestrator.continuous_improvement_cycle(
            codebase_path=self.project_path,
            improvement_targets=improvement_targets
        )
    
    async def get_strategic_advice(self, question: str):
        """Get strategic advice from MASTERMIND."""
        
        print(f"🧠 Getting strategic advice: {question}")
        
        context = {
            "project_path": self.project_path,
            "question": question,
            "current_state": self._analyze_current_state()
        }
        
        return await self.orchestrator.collaborative_problem_solving(
            problem_statement=question,
            complexity_level="medium"
        )
    
    def _detect_tech_stack(self):
        """Detect project's technology stack."""
        project_path = Path(self.project_path)
        
        # Check for common files
        if (project_path / "requirements.txt").exists():
            return "Python (FastAPI/Django/Flask)"
        elif (project_path / "package.json").exists():
            return "Node.js (Express/React/Vue)"
        elif (project_path / "Cargo.toml").exists():
            return "Rust"
        elif (project_path / "go.mod").exists():
            return "Go"
        else:
            return "Unknown (will use Python FastAPI)"
    
    def _analyze_current_state(self):
        """Analyze current project state."""
        return {
            "tech_stack": self._detect_tech_stack(),
            "project_size": "medium",  # Would analyze actual size
            "complexity": "medium"     # Would analyze actual complexity
        }

# Example usage functions
async def example_add_api_endpoint():
    """Example: Add API endpoint to existing project."""
    
    assistant = ProjectAssistant("/home/scottschweizer/TradeKnowledge")
    
    result = await assistant.add_feature(
        "Add a REST API endpoint for real-time search with autocomplete and caching"
    )
    
    print("✅ Feature added!")
    print(f"🎯 Quality: {result['session_results']['metrics'].quality_amplification:.1f}/10")

async def example_improve_performance():
    """Example: Improve performance of existing code."""
    
    assistant = ProjectAssistant("/home/scottschweizer/TradeKnowledge")
    
    result = await assistant.improve_code_quality(
        focus_areas=["performance", "database_optimization"]
    )
    
    print("✅ Performance improvements identified!")
    print(f"📊 Opportunities: {len(result['improvement_opportunities'])}")

async def example_strategic_decision():
    """Example: Get help with strategic decision."""
    
    assistant = ProjectAssistant("/home/scottschweizer/TradeKnowledge")
    
    question = """
    Our API is getting slow with 50k+ daily users. Should we:
    1. Add caching (Redis)
    2. Switch to microservices
    3. Optimize database queries
    4. Use a CDN
    
    We have 3 developers and moderate budget.
    """
    
    advice = await assistant.get_strategic_advice(question)
    
    print("🧠 Strategic recommendation received!")
    print(f"💡 Solution quality: {advice['solution_quality']['overall_score']:.1f}/10")

# Interactive menu
async def interactive_integration():
    """Interactive integration with your project."""
    
    print("🔗 INTEGRATE AGENTS WITH YOUR PROJECT")
    print("=" * 40)
    
    project_path = input("Enter your project path (or press Enter for current TradeKnowledge): ").strip()
    if not project_path:
        project_path = "/home/scottschweizer/TradeKnowledge"
    
    assistant = ProjectAssistant(project_path)
    
    print(f"\n📁 Project: {project_path}")
    print(f"🛠️  Detected: {assistant._detect_tech_stack()}")
    
    while True:
        print("\nWhat would you like to do?")
        print("1. 🆕 Add a new feature")
        print("2. 🔧 Improve code quality")
        print("3. 🧠 Get strategic advice")
        print("4. 📊 Analyze current codebase")
        print("5. 🚪 Exit")
        
        choice = input("\nChoose 1-5: ").strip()
        
        if choice == "1":
            feature = input("Describe the feature to add: ")
            await assistant.add_feature(feature)
            
        elif choice == "2":
            print("Focus areas: performance, security, maintainability, testing")
            areas = input("Enter focus areas (comma-separated, or Enter for all): ").strip()
            focus_areas = [a.strip() for a in areas.split(",")] if areas else None
            await assistant.improve_code_quality(focus_areas)
            
        elif choice == "3":
            question = input("What strategic question do you have? ")
            await assistant.get_strategic_advice(question)
            
        elif choice == "4":
            result = await assistant.improve_code_quality()
            print(f"📊 Analysis complete: {len(result['improvement_opportunities'])} opportunities found")
            
        elif choice == "5":
            break
        
        print("\n" + "="*40)

if __name__ == "__main__":
    print("Choose integration example:")
    print("1. 🆕 Add API endpoint")
    print("2. 🔧 Improve performance") 
    print("3. 🧠 Strategic decision help")
    print("4. 🔗 Interactive integration")
    
    choice = input("\nChoose 1-4: ").strip()
    
    if choice == "1":
        asyncio.run(example_add_api_endpoint())
    elif choice == "2":
        asyncio.run(example_improve_performance())
    elif choice == "3":
        asyncio.run(example_strategic_decision())
    elif choice == "4":
        asyncio.run(interactive_integration())
    else:
        print("Invalid choice!")