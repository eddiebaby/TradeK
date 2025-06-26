#!/usr/bin/env python3
"""
Ollama Modelfile Manager for Agent Trio
Creates and manages specialized agent models for optimal token efficiency
"""

import subprocess
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelfileManager:
    """Manages custom Ollama models for agent trio optimization"""
    
    def __init__(self, modelfiles_dir: str = None):
        self.modelfiles_dir = Path(modelfiles_dir or Path(__file__).parent / "modelfiles")
        self.ollama_url = "http://localhost:11434"
        
        # Agent model definitions
        self.agent_models = {
            "researcher": {
                "modelfile": "ResearcherAgent.modelfile",
                "base_model": "llama2:13b", 
                "custom_name": "researcher-agent:latest",
                "description": "Specialized for intelligence gathering and analysis"
            },
            "mastermind": {
                "modelfile": "MastermindAgent.modelfile", 
                "base_model": "mixtral:8x7b",
                "custom_name": "mastermind-agent:latest",
                "description": "Optimized for strategic planning and architecture"
            },
            "executor": {
                "modelfile": "ExecutorAgent.modelfile",
                "base_model": "codellama:13b",
                "custom_name": "executor-agent:latest", 
                "description": "Tuned for implementation and DevOps tasks"
            }
        }
    
    def check_ollama_status(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of currently available models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    def check_base_models(self) -> Dict[str, bool]:
        """Check which base models are available"""
        available_models = self.get_available_models()
        
        base_model_status = {}
        for agent, config in self.agent_models.items():
            base_model = config["base_model"]
            base_model_status[base_model] = base_model in available_models
        
        return base_model_status
    
    def install_missing_base_models(self) -> Dict[str, bool]:
        """Install any missing base models"""
        logger.info("🔍 Checking for missing base models...")
        
        base_status = self.check_base_models()
        installation_results = {}
        
        for base_model, is_available in base_status.items():
            if not is_available:
                logger.info(f"📦 Installing missing base model: {base_model}")
                try:
                    result = subprocess.run(
                        ["ollama", "pull", base_model],
                        capture_output=True,
                        text=True,
                        timeout=600  # 10 minutes max
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"✅ Successfully installed {base_model}")
                        installation_results[base_model] = True
                    else:
                        logger.error(f"❌ Failed to install {base_model}: {result.stderr}")
                        installation_results[base_model] = False
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"❌ Timeout installing {base_model}")
                    installation_results[base_model] = False
                except Exception as e:
                    logger.error(f"❌ Error installing {base_model}: {e}")
                    installation_results[base_model] = False
            else:
                logger.info(f"✅ Base model already available: {base_model}")
                installation_results[base_model] = True
        
        return installation_results
    
    def create_agent_model(self, agent_name: str) -> bool:
        """Create custom model for specific agent"""
        if agent_name not in self.agent_models:
            logger.error(f"❌ Unknown agent: {agent_name}")
            return False
        
        config = self.agent_models[agent_name]
        modelfile_path = self.modelfiles_dir / config["modelfile"]
        
        if not modelfile_path.exists():
            logger.error(f"❌ Modelfile not found: {modelfile_path}")
            return False
        
        logger.info(f"🔧 Creating custom model for {agent_name.upper()} agent...")
        logger.info(f"   Base: {config['base_model']}")
        logger.info(f"   Target: {config['custom_name']}")
        
        try:
            # Create model using ollama create command
            result = subprocess.run([
                "ollama", "create", 
                config["custom_name"],
                "-f", str(modelfile_path)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully created {config['custom_name']}")
                return True
            else:
                logger.error(f"❌ Failed to create model: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout creating model for {agent_name}")
            return False
        except Exception as e:
            logger.error(f"❌ Error creating model: {e}")
            return False
    
    def create_all_agent_models(self) -> Dict[str, bool]:
        """Create all custom agent models"""
        logger.info("🚀 Creating all custom agent models...")
        
        results = {}
        for agent_name in self.agent_models.keys():
            results[agent_name] = self.create_agent_model(agent_name)
        
        return results
    
    def test_agent_model(self, agent_name: str) -> Dict[str, Any]:
        """Test a custom agent model with a sample prompt"""
        if agent_name not in self.agent_models:
            return {"success": False, "error": "Unknown agent"}
        
        config = self.agent_models[agent_name]
        model_name = config["custom_name"]
        
        # Agent-specific test prompts
        test_prompts = {
            "researcher": "Analyze the cybersecurity implications of AI model deployment in production environments.",
            "mastermind": "Design a microservices architecture for a real-time trading platform that handles 10,000 TPS.",
            "executor": "Implement a Python function that monitors system performance and alerts on anomalies."
        }
        
        prompt = test_prompts.get(agent_name, "Test the model functionality")
        
        logger.info(f"🧪 Testing {agent_name} agent model...")
        
        try:
            # Test using API call
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 500}  # Shorter for testing
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                logger.info(f"✅ {agent_name} model test successful")
                logger.info(f"   Response time: {response_time:.2f}s")
                logger.info(f"   Content length: {len(content)} chars")
                
                return {
                    "success": True,
                    "response_time": response_time,
                    "content_length": len(content),
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                }
            else:
                logger.error(f"❌ {agent_name} model test failed: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"❌ {agent_name} model test error: {e}")
            return {"success": False, "error": str(e)}
    
    def benchmark_token_efficiency(self) -> Dict[str, Any]:
        """Compare token efficiency between base and custom models"""
        logger.info("📊 Benchmarking token efficiency...")
        
        # Standard test prompt
        test_prompt = "Explain the key considerations for implementing a secure API gateway in a microservices architecture."
        
        results = {
            "test_prompt": test_prompt,
            "base_models": {},
            "custom_models": {},
            "efficiency_gains": {}
        }
        
        # Test base models
        base_models = ["llama2:13b", "mixtral:8x7b", "codellama:13b"]
        for model in base_models:
            if model in self.get_available_models():
                result = self._test_model_response(model, test_prompt)
                results["base_models"][model] = result
        
        # Test custom models
        for agent_name, config in self.agent_models.items():
            model_name = config["custom_name"]
            if model_name in self.get_available_models():
                result = self._test_model_response(model_name, test_prompt)
                results["custom_models"][agent_name] = result
                
                # Calculate efficiency gain
                base_model = config["base_model"]
                if base_model in results["base_models"]:
                    base_length = results["base_models"][base_model].get("response_length", 0)
                    custom_length = result.get("response_length", 0)
                    
                    if base_length > 0:
                        efficiency = (base_length - custom_length) / base_length * 100
                        results["efficiency_gains"][agent_name] = {
                            "base_length": base_length,
                            "custom_length": custom_length,
                            "token_reduction_percent": efficiency
                        }
        
        return results
    
    def _test_model_response(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Test a single model response"""
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 1000}
            }
            
            start_time = time.time()
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=60)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                return {
                    "success": True,
                    "response_time": response_time,
                    "response_length": len(content),
                    "estimated_tokens": len(content.split()) * 1.3  # Rough token estimate
                }
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about all models"""
        available_models = self.get_available_models()
        
        info = {
            "ollama_status": self.check_ollama_status(),
            "available_models": available_models,
            "base_models": self.check_base_models(),
            "custom_models": {},
            "agent_configurations": self.agent_models
        }
        
        # Check which custom models exist
        for agent_name, config in self.agent_models.items():
            custom_name = config["custom_name"]
            info["custom_models"][agent_name] = {
                "exists": custom_name in available_models,
                "name": custom_name,
                "base_model": config["base_model"]
            }
        
        return info
    
    def setup_all_models(self) -> Dict[str, Any]:
        """Complete setup of all agent models"""
        logger.info("🚀 Starting complete agent model setup...")
        
        setup_results = {
            "ollama_check": False,
            "base_models": {},
            "custom_models": {},
            "tests": {},
            "overall_success": False
        }
        
        # Check Ollama
        if not self.check_ollama_status():
            logger.error("❌ Ollama is not running")
            return setup_results
        
        setup_results["ollama_check"] = True
        logger.info("✅ Ollama is running")
        
        # Install base models
        setup_results["base_models"] = self.install_missing_base_models()
        
        # Create custom models
        setup_results["custom_models"] = self.create_all_agent_models()
        
        # Test each custom model
        for agent_name in self.agent_models.keys():
            if setup_results["custom_models"].get(agent_name, False):
                setup_results["tests"][agent_name] = self.test_agent_model(agent_name)
        
        # Calculate overall success
        successful_models = sum(1 for success in setup_results["custom_models"].values() if success)
        setup_results["overall_success"] = successful_models >= 2
        
        logger.info(f"\n🎉 Setup complete! {successful_models}/3 agent models created successfully")
        
        return setup_results

def main():
    """Main setup and testing function"""
    manager = ModelfileManager()
    
    # Run complete setup
    results = manager.setup_all_models()
    
    # Show results
    print("\n" + "="*50)
    print("🤖 AGENT MODEL SETUP RESULTS")
    print("="*50)
    
    for agent, success in results["custom_models"].items():
        status = "✅" if success else "❌"
        print(f"{status} {agent.upper()} Agent: {'Ready' if success else 'Failed'}")
    
    # Run benchmarks if models were created successfully
    if results["overall_success"]:
        print("\n📊 Running token efficiency benchmarks...")
        benchmarks = manager.benchmark_token_efficiency()
        
        print("\n🎯 TOKEN EFFICIENCY RESULTS:")
        for agent, gains in benchmarks.get("efficiency_gains", {}).items():
            reduction = gains.get("token_reduction_percent", 0)
            print(f"  {agent.upper()}: {reduction:.1f}% token reduction")
    
    print(f"\n🎯 Overall Success: {'YES' if results['overall_success'] else 'NO'}")
    
    return results

if __name__ == "__main__":
    main()