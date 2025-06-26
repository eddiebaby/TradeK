#!/usr/bin/env python3
"""
Ollama Integration Module for Agent Trio
Provides enhanced Ollama client with agent-specific optimizations
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests
import yaml

logger = logging.getLogger(__name__)

class OllamaAgentClient:
    """Enhanced Ollama client optimized for agent trio workflows"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 120
        
        # Agent-specific model preferences (custom models preferred)
        self.agent_models = {
            "Researcher": {
                "custom": "researcher-agent:latest",
                "primary": "llama2:13b",
                "code_analysis": "codellama:13b", 
                "complex_reasoning": "mixtral:8x7b"
            },
            "Mastermind": {
                "custom": "mastermind-agent:latest",
                "primary": "mixtral:8x7b",
                "planning": "llama2:13b",
                "architecture": "codellama:13b"
            },
            "Executor": {
                "custom": "executor-agent:latest",
                "primary": "codellama:13b",
                "general": "llama2:13b",
                "complex_code": "mixtral:8x7b"
            }
        }
        
        # Performance tracking
        self.performance_cache = {}
        
    def is_healthy(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[Dict]:
        """Get detailed list of available models"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return response.json().get('models', [])
            return []
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    def select_model_for_task(self, agent_name: str, operation: str, complexity: float) -> str:
        """Select optimal model for specific agent task (prefer custom models)"""
        agent_models = self.agent_models.get(agent_name, {})
        available_models = [m["name"] for m in self.get_available_models()]
        
        # Try custom model first (most token-efficient)
        custom_model = agent_models.get("custom")
        if custom_model and custom_model in available_models:
            return custom_model
        
        # Fallback to operation-specific model selection
        if "code" in operation.lower() or "implementation" in operation.lower():
            return agent_models.get("code_analysis", "codellama:13b")
        elif complexity > 0.7:
            return agent_models.get("complex_reasoning", "mixtral:8x7b")
        elif "architecture" in operation.lower() or "design" in operation.lower():
            return agent_models.get("architecture", "mixtral:8x7b")
        else:
            return agent_models.get("primary", "llama2:13b")
    
    async def generate_completion(self, 
                                prompt: str, 
                                model: str = "llama2:13b",
                                agent_name: str = "Unknown",
                                operation: str = "general",
                                max_tokens: int = 2000,
                                temperature: float = 0.7) -> Dict[str, Any]:
        """Generate completion with agent-specific optimizations"""
        
        # Check if using custom model (already has optimized prompts)
        agent_models = self.agent_models.get(agent_name, {})
        is_custom_model = model == agent_models.get("custom")
        
        # Use prompt as-is for custom models, optimize for base models
        optimized_prompt = prompt if is_custom_model else self._optimize_prompt_for_agent(prompt, agent_name, operation)
        
        payload = {
            "model": model,
            "prompt": optimized_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            response_time = time.time() - start_time
            
            # Post-process response for agent
            processed_content = self._post_process_response(
                result.get("response", ""), 
                agent_name, 
                operation
            )
            
            # Track performance
            self._track_performance(model, agent_name, operation, response_time)
            
            return {
                "content": processed_content,
                "model": model,
                "agent": agent_name,
                "operation": operation,
                "response_time": response_time,
                "tokens_used": self._estimate_tokens(processed_content),
                "cost": 0.0,  # Local models are free
                "source": "ollama",
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Ollama generation failed for {agent_name}: {e}")
            return {
                "content": "",
                "model": model,
                "agent": agent_name,
                "operation": operation,
                "error": str(e),
                "success": False
            }
    
    def _optimize_prompt_for_agent(self, prompt: str, agent_name: str, operation: str) -> str:
        """Optimize prompt based on agent role and operation"""
        
        agent_contexts = {
            "Researcher": {
                "role": "You are an expert researcher and intelligence analyst",
                "focus": "Provide thorough, well-researched information with citations and analysis",
                "style": "analytical, comprehensive, evidence-based"
            },
            "Mastermind": {
                "role": "You are a strategic architect and system designer", 
                "focus": "Design robust, scalable solutions with clear architectural reasoning",
                "style": "strategic, systematic, forward-thinking"
            },
            "Executor": {
                "role": "You are an implementation expert and DevOps engineer",
                "focus": "Provide practical, actionable implementation with best practices",
                "style": "practical, efficient, production-ready"
            }
        }
        
        context = agent_contexts.get(agent_name, agent_contexts["Researcher"])
        
        optimized_prompt = f"""
{context['role']}. {context['focus']}.

Operation: {operation}
Style: {context['style']}

Task: {prompt}

Please provide a {context['style']} response that directly addresses the task requirements.
"""
        
        return optimized_prompt.strip()
    
    def _post_process_response(self, content: str, agent_name: str, operation: str) -> str:
        """Post-process response based on agent requirements"""
        
        # Agent-specific formatting
        if agent_name == "Researcher":
            # Ensure research includes sources and analysis
            if "analysis:" not in content.lower() and len(content) > 200:
                content += "\n\n## Analysis Summary\nBased on the above information, key insights include the main findings and their implications for the requested research area."
                
        elif agent_name == "Mastermind":
            # Ensure strategic output includes architecture considerations
            if "architecture" in operation.lower() and "components:" not in content.lower():
                content += "\n\n## Architectural Components\nKey system components and their interactions should be considered in the implementation."
                
        elif agent_name == "Executor":
            # Ensure implementation includes practical steps
            if "implementation" in operation.lower() and "steps:" not in content.lower():
                content += "\n\n## Implementation Steps\n1. Detailed implementation steps should be provided\n2. Include testing and validation procedures\n3. Consider deployment and monitoring requirements"
        
        return content
    
    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content"""
        # Rough estimation: ~4 characters per token
        return len(content) // 4
    
    def _track_performance(self, model: str, agent: str, operation: str, response_time: float):
        """Track model performance metrics"""
        key = f"{model}_{agent}_{operation}"
        
        if key not in self.performance_cache:
            self.performance_cache[key] = {
                "total_requests": 0,
                "total_time": 0.0,
                "avg_response_time": 0.0
            }
        
        stats = self.performance_cache[key]
        stats["total_requests"] += 1
        stats["total_time"] += response_time
        stats["avg_response_time"] = stats["total_time"] / stats["total_requests"]
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            "cache": self.performance_cache,
            "available_models": [m["name"] for m in self.get_available_models()],
            "health_status": self.is_healthy()
        }
    
    async def warmup_models(self, models: List[str] = None) -> Dict[str, bool]:
        """Warm up specified models to improve response times"""
        if models is None:
            models = ["llama2:13b", "codellama:13b", "mixtral:8x7b"]
        
        warmup_results = {}
        
        for model in models:
            try:
                # Send a simple warmup prompt
                warmup_prompt = "Hello, this is a warmup request. Please respond with 'Ready'."
                
                result = await self.generate_completion(
                    prompt=warmup_prompt,
                    model=model,
                    max_tokens=10,
                    temperature=0.0
                )
                
                warmup_results[model] = result.get("success", False)
                logger.info(f"Warmed up model {model}: {warmup_results[model]}")
                
            except Exception as e:
                warmup_results[model] = False
                logger.warning(f"Failed to warm up {model}: {e}")
        
        return warmup_results

# Singleton instance
ollama_client = OllamaAgentClient()

# Convenience functions for agents
async def researcher_completion(prompt: str, operation: str = "research", **kwargs) -> Dict:
    """Generate completion optimized for Researcher agent"""
    complexity = kwargs.pop("complexity", 0.5)  # Remove from kwargs
    model = ollama_client.select_model_for_task("Researcher", operation, complexity)
    
    return await ollama_client.generate_completion(
        prompt=prompt,
        model=model,
        agent_name="Researcher", 
        operation=operation,
        **kwargs
    )

async def mastermind_completion(prompt: str, operation: str = "strategy", **kwargs) -> Dict:
    """Generate completion optimized for Mastermind agent"""
    complexity = kwargs.pop("complexity", 0.7)  # Remove from kwargs
    model = ollama_client.select_model_for_task("Mastermind", operation, complexity)
    
    return await ollama_client.generate_completion(
        prompt=prompt,
        model=model,
        agent_name="Mastermind",
        operation=operation,
        **kwargs
    )

async def executor_completion(prompt: str, operation: str = "implementation", **kwargs) -> Dict:
    """Generate completion optimized for Executor agent"""
    complexity = kwargs.pop("complexity", 0.4)  # Remove from kwargs
    model = ollama_client.select_model_for_task("Executor", operation, complexity)
    
    return await ollama_client.generate_completion(
        prompt=prompt,
        model=model,
        agent_name="Executor",
        operation=operation,
        **kwargs
    )