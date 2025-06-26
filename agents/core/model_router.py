#!/usr/bin/env python3
"""
Hybrid Model Router for Agent Trio
Routes tasks between local Ollama models and cloud models based on complexity and cost optimization
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Literal
from dataclasses import dataclass
from enum import Enum
import requests
import logging

from influx_blackboard import log_performance

logger = logging.getLogger(__name__)

class ModelChoice(Enum):
    LOCAL_OLLAMA = "local_ollama"
    CLOUD_PREMIUM = "cloud_premium"
    HYBRID_FALLBACK = "hybrid_fallback"

class TaskComplexity(Enum):
    SIMPLE = "simple"      # 0.0-0.3
    MODERATE = "moderate"  # 0.3-0.6
    COMPLEX = "complex"    # 0.6-0.8
    CRITICAL = "critical"  # 0.8-1.0

@dataclass
class TaskContext:
    agent_name: str
    operation: str
    description: str
    requires_web_search: bool = False
    requires_realtime_data: bool = False
    business_impact: str = "low"  # low, medium, high, critical
    expected_output_length: int = 500
    domain_complexity: str = "standard"  # standard, technical, novel

@dataclass
class ModelConfig:
    name: str
    type: str  # ollama, claude, openai
    endpoint: str
    max_tokens: int
    cost_per_token: float
    avg_response_time: float
    quality_score: float

class OllamaClient:
    """Client for interacting with local Ollama models"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = self._get_available_models()
        
    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def complete(self, prompt: str, model: str = "llama2:13b", 
                      max_tokens: int = 2000) -> Dict:
        """Generate completion using Ollama model"""
        if not self.is_available():
            raise Exception("Ollama not available")
            
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            response_time = time.time() - start_time
            
            return {
                "content": result.get("response", ""),
                "model": model,
                "response_time": response_time,
                "tokens_used": len(result.get("response", "").split()) * 1.3,  # Estimate
                "cost": 0.0,  # Local models are free
                "source": "ollama"
            }
            
        except Exception as e:
            logger.error(f"Ollama completion failed: {e}")
            raise

class HybridModelRouter:
    """Smart router for choosing between local and cloud models"""
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.model_configs = self._load_model_configs()
        self.usage_stats = {}
        
    def _load_model_configs(self) -> Dict[str, ModelConfig]:
        """Load model configurations"""
        return {
            # Ollama models
            "codellama:13b": ModelConfig(
                name="codellama:13b",
                type="ollama",
                endpoint="localhost:11434",
                max_tokens=4000,
                cost_per_token=0.0,
                avg_response_time=3.5,
                quality_score=0.85
            ),
            "llama2:13b": ModelConfig(
                name="llama2:13b", 
                type="ollama",
                endpoint="localhost:11434",
                max_tokens=4000,
                cost_per_token=0.0,
                avg_response_time=3.0,
                quality_score=0.82
            ),
            "mixtral:8x7b": ModelConfig(
                name="mixtral:8x7b",
                type="ollama", 
                endpoint="localhost:11434",
                max_tokens=4000,
                cost_per_token=0.0,
                avg_response_time=8.0,
                quality_score=0.90
            ),
            # Cloud models
            "claude-3-sonnet": ModelConfig(
                name="claude-3-sonnet",
                type="claude",
                endpoint="api.anthropic.com",
                max_tokens=8000,
                cost_per_token=0.00003,
                avg_response_time=2.0,
                quality_score=0.95
            )
        }
    
    def assess_complexity(self, task: TaskContext) -> float:
        """Assess task complexity (0.0-1.0)"""
        complexity_score = 0.0
        
        # Base complexity from operation type  
        operation_complexity = {
            "security_intelligence": 0.7,
            "market_intelligence": 0.6,
            "technical_analysis": 0.5,
            "architecture_design": 0.8,
            "strategic_planning": 0.9,
            "quality_strategy": 0.6,
            "implementation": 0.4,
            "testing": 0.3,
            "deployment": 0.5,
            "code_review": 0.3,
            "documentation": 0.2
        }
        complexity_score += operation_complexity.get(task.operation, 0.5)
        
        # Adjust for requirements
        if task.requires_web_search:
            complexity_score += 0.3
        if task.requires_realtime_data:
            complexity_score += 0.2
        if task.business_impact == "critical":
            complexity_score += 0.2
        if task.domain_complexity == "novel":
            complexity_score += 0.3
            
        # Adjust for output length
        if task.expected_output_length > 2000:
            complexity_score += 0.1
            
        return min(complexity_score, 1.0)
    
    def estimate_tokens(self, task: TaskContext) -> int:
        """Estimate token usage for task"""
        base_tokens = len(task.description.split()) * 1.3
        
        # Output length estimation
        output_tokens = task.expected_output_length / 4  # ~4 chars per token
        
        # Operation-specific multipliers
        operation_multipliers = {
            "security_intelligence": 2.5,
            "market_intelligence": 2.0,
            "technical_analysis": 1.8,
            "architecture_design": 3.0,
            "strategic_planning": 2.8,
            "implementation": 1.5,
            "testing": 1.2,
            "code_review": 1.3,
            "documentation": 1.1
        }
        
        multiplier = operation_multipliers.get(task.operation, 1.5)
        return int((base_tokens + output_tokens) * multiplier)
    
    def route_task(self, task: TaskContext) -> ModelChoice:
        """Determine best model for task"""
        complexity = self.assess_complexity(task)
        estimated_tokens = self.estimate_tokens(task)
        
        # Check if Ollama is available
        ollama_available = self.ollama_client.is_available()
        
        # Hard requirements for cloud models
        if (task.requires_web_search or 
            task.requires_realtime_data or 
            task.business_impact == "critical"):
            return ModelChoice.CLOUD_PREMIUM
            
        # Route to local if simple/moderate and Ollama available
        if (complexity <= 0.6 and 
            estimated_tokens < 3000 and 
            ollama_available):
            return ModelChoice.LOCAL_OLLAMA
            
        # Complex but suitable for powerful local models
        if (complexity <= 0.8 and 
            estimated_tokens < 4000 and 
            ollama_available and
            "mixtral:8x7b" in self.ollama_client.available_models):
            return ModelChoice.LOCAL_OLLAMA
            
        # Default to cloud for complex tasks
        return ModelChoice.CLOUD_PREMIUM
    
    def select_model(self, task: TaskContext, choice: ModelChoice) -> str:
        """Select specific model based on routing choice"""
        if choice == ModelChoice.LOCAL_OLLAMA:
            # Select best local model for task type
            if task.operation in ["implementation", "testing", "code_review"]:
                return "codellama:13b"
            elif task.assess_complexity(task) > 0.6:
                return "mixtral:8x7b" if "mixtral:8x7b" in self.ollama_client.available_models else "llama2:13b"
            else:
                return "llama2:13b"
        else:
            return "claude-3-sonnet"
    
    async def execute_task(self, task: TaskContext, prompt: str) -> Dict:
        """Execute task with selected model and fallback capability"""
        choice = self.route_task(task)
        model = self.select_model(task, choice)
        
        start_time = time.time()
        
        try:
            if choice == ModelChoice.LOCAL_OLLAMA:
                result = await self.ollama_client.complete(prompt, model)
                result["routing_choice"] = choice.value
                result["cost_savings"] = self._calculate_savings(task, result)
            else:
                # Cloud model execution would go here
                # For now, return placeholder
                result = {
                    "content": f"[CLOUD MODEL EXECUTION PLACEHOLDER for {model}]\n{prompt}",
                    "model": model,
                    "response_time": 2.0,
                    "tokens_used": self.estimate_tokens(task),
                    "cost": self.estimate_tokens(task) * 0.00003,
                    "source": "cloud",
                    "routing_choice": choice.value,
                    "cost_savings": 0.0
                }
                
            # Log performance metrics
            await self._log_routing_metrics(task, result)
            return result
            
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            
            # Intelligent fallback
            if choice == ModelChoice.LOCAL_OLLAMA:
                logger.info("Falling back to cloud model")
                # In production, this would call cloud model
                return {
                    "content": f"[FALLBACK EXECUTION]\nOriginal task: {prompt}\nError: {str(e)}",
                    "model": "claude-3-sonnet",
                    "response_time": 2.0,
                    "tokens_used": self.estimate_tokens(task),
                    "cost": self.estimate_tokens(task) * 0.00003,
                    "source": "cloud_fallback",
                    "routing_choice": "fallback",
                    "error": str(e)
                }
            else:
                raise
    
    def _calculate_savings(self, task: TaskContext, result: Dict) -> float:
        """Calculate cost savings from using local model"""
        cloud_cost = result["tokens_used"] * 0.00003  # Claude pricing
        local_cost = 0.0
        return cloud_cost - local_cost
    
    async def _log_routing_metrics(self, task: TaskContext, result: Dict):
        """Log routing decisions and performance to InfluxDB"""
        try:
            await log_performance(
                agent_name=task.agent_name,
                operation=task.operation,
                success_rate=1.0 if "error" not in result else 0.0,
                avg_tokens=result["tokens_used"],
                efficiency_score=1.0 / max(result["response_time"], 0.1),
                cost_savings=result.get("cost_savings", 0.0),
                model_used=result["model"],
                routing_choice=result.get("routing_choice", "unknown")
            )
        except Exception as e:
            logger.warning(f"Failed to log routing metrics: {e}")
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics and cost savings"""
        return {
            "ollama_available": self.ollama_client.is_available(),
            "available_models": self.ollama_client.available_models,
            "routing_decisions": self.usage_stats,
            "estimated_monthly_savings": self._calculate_monthly_savings()
        }
    
    def _calculate_monthly_savings(self) -> Dict:
        """Calculate estimated monthly cost savings"""
        # Placeholder calculation - would use actual usage data
        return {
            "current_monthly_cost": 300.0,
            "projected_cost_with_ollama": 88.0,
            "monthly_savings": 212.0,
            "savings_percentage": 70.7
        }

# Global router instance
model_router = HybridModelRouter()

async def route_and_execute(agent_name: str, operation: str, prompt: str, **kwargs) -> Dict:
    """Main entry point for routing and executing tasks"""
    task = TaskContext(
        agent_name=agent_name,
        operation=operation,
        description=prompt[:500],  # First 500 chars for analysis
        **kwargs
    )
    
    return await model_router.execute_task(task, prompt)