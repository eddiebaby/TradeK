#!/usr/bin/env python3
"""
Enhanced Agent Wrapper with Ollama Integration
Provides upgraded agent interfaces with hybrid local/cloud model routing
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import argparse

# Add core modules to path
sys.path.append(str(Path(__file__).parent / "core"))
sys.path.append(str(Path(__file__).parent))

from model_router import route_and_execute, TaskContext, ModelChoice
from ollama_integration import ollama_client, researcher_completion, mastermind_completion, executor_completion
from influx_blackboard import write_task, update_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAgent:
    """Enhanced agent with Ollama integration and intelligent routing"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.session_stats = {
            "total_requests": 0,
            "local_model_requests": 0,
            "cloud_model_requests": 0,
            "total_cost_savings": 0.0,
            "total_response_time": 0.0
        }
    
    async def process_request(self, 
                            prompt: str,
                            operation: str = "general",
                            force_local: bool = False,
                            force_cloud: bool = False,
                            **kwargs) -> Dict[str, Any]:
        """Process a request with intelligent model routing"""
        
        self.session_stats["total_requests"] += 1
        
        # Create task context
        task_context = TaskContext(
            agent_name=self.agent_name,
            operation=operation,
            description=prompt,
            **kwargs
        )
        
        # Log task start
        task_id = await write_task(self.agent_name, operation, prompt)
        await update_status(task_id, "processing")
        
        try:
            # Choose execution method
            if force_local and ollama_client.is_healthy():
                result = await self._execute_local(prompt, operation, **kwargs)
            elif force_cloud:
                result = await self._execute_cloud(prompt, operation, **kwargs)
            else:
                # Use intelligent routing
                result = await route_and_execute(
                    agent_name=self.agent_name,
                    operation=operation,
                    prompt=prompt,
                    **kwargs
                )
            
            # Update session statistics
            self._update_stats(result)
            
            # Log task completion
            await update_status(task_id, "completed")
            
            # Add session info to result
            result["session_stats"] = self.session_stats.copy()
            result["task_id"] = task_id
            
            return result
            
        except Exception as e:
            await update_status(task_id, "failed")
            logger.error(f"Request processing failed: {e}")
            raise
    
    async def _execute_local(self, prompt: str, operation: str, **kwargs) -> Dict:
        """Execute using local Ollama models"""
        if self.agent_name == "Researcher":
            result = await researcher_completion(prompt, operation, **kwargs)
        elif self.agent_name == "Mastermind":
            result = await mastermind_completion(prompt, operation, **kwargs)
        elif self.agent_name == "Executor":
            result = await executor_completion(prompt, operation, **kwargs)
        else:
            raise ValueError(f"Unknown agent: {self.agent_name}")
        
        result["routing_choice"] = ModelChoice.LOCAL_OLLAMA.value
        result["cost_savings"] = self._calculate_cloud_equivalent_cost(result)
        return result
    
    async def _execute_cloud(self, prompt: str, operation: str, **kwargs) -> Dict:
        """Execute using cloud models (placeholder)"""
        # In production, this would call actual cloud APIs
        return {
            "content": f"[CLOUD EXECUTION PLACEHOLDER]\nAgent: {self.agent_name}\nOperation: {operation}\nPrompt: {prompt}",
            "model": "claude-3-sonnet",
            "response_time": 2.0,
            "tokens_used": len(prompt.split()) * 3,
            "cost": len(prompt.split()) * 3 * 0.00003,
            "source": "cloud",
            "routing_choice": ModelChoice.CLOUD_PREMIUM.value,
            "cost_savings": 0.0
        }
    
    def _calculate_cloud_equivalent_cost(self, local_result: Dict) -> float:
        """Calculate what this request would have cost with cloud models"""
        tokens = local_result.get("tokens_used", 0)
        cloud_cost_per_token = 0.00003  # Claude pricing
        return tokens * cloud_cost_per_token
    
    def _update_stats(self, result: Dict):
        """Update session statistics"""
        self.session_stats["total_response_time"] += result.get("response_time", 0)
        
        if result.get("source") == "ollama":
            self.session_stats["local_model_requests"] += 1
        else:
            self.session_stats["cloud_model_requests"] += 1
        
        self.session_stats["total_cost_savings"] += result.get("cost_savings", 0)
    
    def get_session_summary(self) -> Dict:
        """Get summary of current session performance"""
        total_requests = self.session_stats["total_requests"]
        if total_requests == 0:
            return {"message": "No requests processed in this session"}
        
        avg_response_time = self.session_stats["total_response_time"] / total_requests
        local_percentage = (self.session_stats["local_model_requests"] / total_requests) * 100
        
        return {
            "agent": self.agent_name,
            "total_requests": total_requests,
            "local_model_usage": f"{local_percentage:.1f}%",
            "average_response_time": f"{avg_response_time:.2f}s",
            "total_cost_savings": f"${self.session_stats['total_cost_savings']:.4f}",
            "ollama_available": ollama_client.is_healthy()
        }

class InteractiveAgentSession:
    """Interactive session handler for enhanced agents"""
    
    def __init__(self):
        self.agents = {
            "researcher": EnhancedAgent("Researcher"),
            "mastermind": EnhancedAgent("Mastermind"),
            "executor": EnhancedAgent("Executor")
        }
        self.current_agent = None
    
    def print_welcome(self):
        """Print welcome message and instructions"""
        print("🤖 Enhanced Agent Trio with Ollama Integration")
        print("=" * 50)
        print("Available agents:")
        print("  🔍 researcher  - Research and intelligence gathering")
        print("  🧠 mastermind  - Strategic planning and architecture")
        print("  ⚡ executor    - Implementation and deployment")
        print()
        print("Commands:")
        print("  /switch <agent>  - Switch to different agent")
        print("  /stats          - Show session statistics")
        print("  /local          - Force next request to use local models")
        print("  /cloud          - Force next request to use cloud models")
        print("  /status         - Show Ollama and system status")
        print("  /help           - Show this help")
        print("  /quit           - Exit session")
        print()
        
        # Check Ollama status
        if ollama_client.is_healthy():
            models = [m["name"] for m in ollama_client.get_available_models()]
            print(f"✅ Ollama available with {len(models)} models")
        else:
            print("⚠️ Ollama not available - will use cloud models only")
        print()
    
    async def run_interactive_session(self):
        """Run interactive agent session"""
        self.print_welcome()
        
        # Default to researcher
        self.current_agent = self.agents["researcher"]
        print("🔍 Selected: Researcher Agent")
        print("Type your research request or /help for commands")
        print()
        
        force_local = False
        force_cloud = False
        
        while True:
            try:
                user_input = input(f"[{self.current_agent.agent_name}] > ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    command = user_input.lower()
                    
                    if command == "/quit" or command == "/exit":
                        print("👋 Goodbye!")
                        break
                    
                    elif command.startswith("/switch "):
                        agent_name = command.split(" ", 1)[1].lower()
                        if agent_name in self.agents:
                            self.current_agent = self.agents[agent_name]
                            print(f"✅ Switched to {self.current_agent.agent_name} Agent")
                        else:
                            print(f"❌ Unknown agent: {agent_name}")
                    
                    elif command == "/stats":
                        for agent in self.agents.values():
                            summary = agent.get_session_summary()
                            if "message" not in summary:
                                print(f"{summary['agent']}: {summary['total_requests']} requests, "
                                     f"{summary['local_model_usage']} local, "
                                     f"{summary['total_cost_savings']} saved")
                    
                    elif command == "/local":
                        force_local = True
                        force_cloud = False
                        print("🏠 Next request will use local models")
                    
                    elif command == "/cloud":
                        force_cloud = True
                        force_local = False
                        print("☁️ Next request will use cloud models")
                    
                    elif command == "/status":
                        if ollama_client.is_healthy():
                            models = ollama_client.get_available_models()
                            print(f"✅ Ollama: {len(models)} models available")
                            for model in models[:3]:  # Show first 3
                                print(f"   • {model['name']}")
                        else:
                            print("❌ Ollama: Not available")
                    
                    elif command == "/help":
                        self.print_welcome()
                    
                    else:
                        print("❌ Unknown command. Type /help for available commands")
                    
                    continue
                
                # Process regular request
                print(f"🔄 Processing with {self.current_agent.agent_name}...")
                
                result = await self.current_agent.process_request(
                    prompt=user_input,
                    operation="interactive",
                    force_local=force_local,
                    force_cloud=force_cloud
                )
                
                # Reset force flags
                force_local = force_cloud = False
                
                # Display result
                print(f"\n📝 Response ({result.get('model', 'unknown')}):")
                print("-" * 40)
                print(result.get("content", "No response"))
                print("-" * 40)
                print(f"⏱️ {result.get('response_time', 0):.1f}s | "
                     f"💰 ${result.get('cost_savings', 0):.4f} saved | "
                     f"🔄 {result.get('routing_choice', 'unknown')}")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Agent Trio with Ollama")
    parser.add_argument("--agent", choices=["researcher", "mastermind", "executor"], 
                       help="Start with specific agent")
    parser.add_argument("--prompt", help="Single prompt to process")
    parser.add_argument("--operation", default="general", help="Operation type")
    parser.add_argument("--local", action="store_true", help="Force local model")
    parser.add_argument("--cloud", action="store_true", help="Force cloud model")
    
    args = parser.parse_args()
    
    if args.prompt:
        # Single request mode
        agent_name = args.agent or "researcher"
        agent = EnhancedAgent(agent_name.capitalize())
        
        result = await agent.process_request(
            prompt=args.prompt,
            operation=args.operation,
            force_local=args.local,
            force_cloud=args.cloud
        )
        
        print(result.get("content", "No response"))
        print(f"\nModel: {result.get('model')} | Time: {result.get('response_time', 0):.1f}s | Savings: ${result.get('cost_savings', 0):.4f}")
    
    else:
        # Interactive mode
        session = InteractiveAgentSession()
        await session.run_interactive_session()

if __name__ == "__main__":
    asyncio.run(main())