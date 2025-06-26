"""
SPARC Agent Compression Integration
Integrates LLMLingua compression with SPARC trio agents for cost optimization
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Protocol, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .llmlingua_service import LLMLinguaService, CompressionConfig, CompressionResult

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """SPARC Agent roles"""
    RESEARCHER = "RESEARCHER"
    MASTERMIND = "MASTERMIND"
    EXECUTOR = "EXECUTOR"


class CompressionStrategy(Enum):
    """Different compression strategies for various scenarios"""
    AGGRESSIVE = "aggressive"      # Maximum compression for cost savings
    BALANCED = "balanced"         # Balance compression and quality
    CONSERVATIVE = "conservative" # Minimal compression, preserve quality
    ADAPTIVE = "adaptive"         # Adapt based on prompt characteristics
    AGENT_OPTIMIZED = "agent_optimized"  # Optimized for specific agent


@dataclass
class AgentPromptContext:
    """Context information for agent prompt compression"""
    agent_role: AgentRole
    task_type: str  # "analysis", "research", "implementation", "handoff"
    priority: str = "medium"  # "low", "medium", "high", "critical"
    domain: str = "trading"  # "trading", "general", "technical"
    preserve_examples: bool = True
    preserve_data: bool = True
    max_compression_ratio: float = 0.3  # Maximum compression allowed
    
    # Agent-specific settings
    researcher_preserve_sources: bool = True
    mastermind_preserve_strategy: bool = True
    executor_preserve_code: bool = True


@dataclass
class AgentCompressionProfile:
    """Compression profile for specific agent type"""
    role: AgentRole
    default_strategy: CompressionStrategy
    compression_configs: Dict[str, CompressionConfig] = field(default_factory=dict)
    preservation_rules: List[str] = field(default_factory=list)
    cost_optimization_priority: float = 0.5  # 0.0 = quality focus, 1.0 = cost focus


class AgentCompressionService:
    """Service for compressing SPARC agent prompts with role-specific optimization"""
    
    def __init__(self, llmlingua_service: LLMLinguaService):
        self.llmlingua_service = llmlingua_service
        self.agent_profiles = self._create_agent_profiles()
        self.compression_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.agent_stats = {
            role.value: {
                "total_compressions": 0,
                "total_tokens_saved": 0,
                "average_compression_ratio": 0.0,
                "cost_savings": 0.0,
                "quality_score": 0.0
            }
            for role in AgentRole
        }
    
    def _create_agent_profiles(self) -> Dict[AgentRole, AgentCompressionProfile]:
        """Create compression profiles for each SPARC agent"""
        profiles = {}
        
        # RESEARCHER Agent Profile
        researcher_configs = {
            "research_task": CompressionConfig(
                compression_ratio=0.7,  # Preserve research context
                force_tokens=[
                    "data", "analysis", "research", "findings", "sources", 
                    "market", "stocks", "trading", "financial", "performance"
                ],
                preserve_instructions=True,
                preserve_question=True
            ),
            "data_analysis": CompressionConfig(
                compression_ratio=0.6,  # More aggressive for pure data
                force_tokens=["data", "metrics", "analysis", "trends", "patterns"],
                preserve_semantic_integrity=True
            ),
            "handoff": CompressionConfig(
                compression_ratio=0.8,  # Preserve handoff context
                force_tokens=["MASTERMIND", "EXECUTOR", "handoff", "results", "context"],
                preserve_instructions=True
            )
        }
        
        profiles[AgentRole.RESEARCHER] = AgentCompressionProfile(
            role=AgentRole.RESEARCHER,
            default_strategy=CompressionStrategy.CONSERVATIVE,
            compression_configs=researcher_configs,
            preservation_rules=[
                "preserve_data_sources",
                "preserve_analysis_context", 
                "preserve_research_methodology"
            ],
            cost_optimization_priority=0.3  # Quality-focused
        )
        
        # MASTERMIND Agent Profile  
        mastermind_configs = {
            "strategy_design": CompressionConfig(
                compression_ratio=0.5,  # Aggressive compression for strategy
                force_tokens=[
                    "strategy", "architecture", "design", "plan", "goals",
                    "trading", "market", "risk", "optimization", "framework"
                ],
                preserve_semantic_integrity=True,
                preserve_instructions=True
            ),
            "analysis_synthesis": CompressionConfig(
                compression_ratio=0.6,  # Balanced for synthesis
                force_tokens=["analysis", "synthesis", "insights", "recommendations"],
                preserve_question=True
            ),
            "orchestration": CompressionConfig(
                compression_ratio=0.7,  # Preserve orchestration context
                force_tokens=["RESEARCHER", "EXECUTOR", "orchestration", "workflow"],
                preserve_instructions=True
            )
        }
        
        profiles[AgentRole.MASTERMIND] = AgentCompressionProfile(
            role=AgentRole.MASTERMIND,
            default_strategy=CompressionStrategy.BALANCED,
            compression_configs=mastermind_configs,
            preservation_rules=[
                "preserve_strategic_context",
                "preserve_architectural_decisions",
                "preserve_orchestration_logic"
            ],
            cost_optimization_priority=0.6  # Balanced approach
        )
        
        # EXECUTOR Agent Profile
        executor_configs = {
            "implementation": CompressionConfig(
                compression_ratio=0.6,  # Balanced for implementation
                force_tokens=[
                    "implementation", "testing", "code", "deployment", "execution",
                    "TDD", "quality", "performance", "monitoring", "metrics"
                ],
                preserve_instructions=True,
                preserve_question=True
            ),
            "testing": CompressionConfig(
                compression_ratio=0.4,  # Conservative for testing contexts
                force_tokens=["testing", "TDD", "quality", "coverage", "validation"],
                preserve_semantic_integrity=True
            ),
            "deployment": CompressionConfig(
                compression_ratio=0.7,  # More aggressive for deployment
                force_tokens=["deployment", "production", "monitoring", "ops"],
                preserve_instructions=True
            )
        }
        
        profiles[AgentRole.EXECUTOR] = AgentCompressionProfile(
            role=AgentRole.EXECUTOR,
            default_strategy=CompressionStrategy.AGENT_OPTIMIZED,
            compression_configs=executor_configs,
            preservation_rules=[
                "preserve_implementation_details",
                "preserve_testing_requirements",
                "preserve_quality_gates"
            ],
            cost_optimization_priority=0.7  # Cost-optimization focused
        )
        
        return profiles
    
    async def compress_agent_prompt(
        self,
        prompt: str,
        context: AgentPromptContext,
        strategy: Optional[CompressionStrategy] = None
    ) -> CompressionResult:
        """
        Compress prompt for specific SPARC agent with optimized settings
        
        Args:
            prompt: The agent prompt to compress
            context: Agent-specific context information
            strategy: Override default compression strategy
        
        Returns:
            CompressionResult with agent-specific optimizations
        """
        profile = self.agent_profiles[context.agent_role]
        strategy = strategy or profile.default_strategy
        
        # Select compression config based on task type
        config = self._select_compression_config(profile, context, strategy)
        
        # Add agent-specific context
        compression_context = {
            "agent_role": context.agent_role.value,
            "task_type": context.task_type,
            "priority": context.priority,
            "domain": context.domain,
            "strategy": strategy.value
        }
        
        # Perform compression
        result = await self.llmlingua_service.compress_prompt(
            prompt, config, compression_context
        )
        
        # Post-process result for agent-specific requirements
        processed_result = await self._post_process_agent_result(
            result, context, profile
        )
        
        # Update agent statistics
        await self._update_agent_stats(context.agent_role, processed_result)
        
        # Log agent-specific metrics
        await self._log_agent_compression(context, processed_result)
        
        return processed_result
    
    async def compress_agent_handoff(
        self,
        from_agent: AgentRole,
        to_agent: AgentRole,
        handoff_data: Dict[str, Any],
        preserve_context: bool = True
    ) -> CompressionResult:
        """
        Compress handoff data between SPARC agents
        
        Args:
            from_agent: Source agent role
            to_agent: Target agent role  
            handoff_data: Data being handed off
            preserve_context: Whether to preserve handoff context
        
        Returns:
            CompressionResult optimized for agent handoffs
        """
        # Create handoff prompt
        handoff_prompt = self._create_handoff_prompt(from_agent, to_agent, handoff_data)
        
        # Create handoff context
        context = AgentPromptContext(
            agent_role=to_agent,  # Optimize for receiving agent
            task_type="handoff",
            priority="high",
            preserve_examples=preserve_context,
            preserve_data=True
        )
        
        # Use conservative strategy for handoffs to preserve information
        strategy = CompressionStrategy.CONSERVATIVE
        
        result = await self.compress_agent_prompt(handoff_prompt, context, strategy)
        
        # Add handoff metadata
        result.compressed_prompt = self._add_handoff_metadata(
            result.compressed_prompt, from_agent, to_agent
        )
        
        return result
    
    async def compress_trio_workflow(
        self,
        workflow_prompt: str,
        workflow_context: Dict[str, Any]
    ) -> Dict[AgentRole, CompressionResult]:
        """
        Compress prompts for entire SPARC trio workflow
        
        Args:
            workflow_prompt: Base workflow prompt
            workflow_context: Context for the entire workflow
        
        Returns:
            Dict mapping each agent role to its compression result
        """
        results = {}
        
        # Create agent-specific contexts
        contexts = {
            AgentRole.RESEARCHER: AgentPromptContext(
                agent_role=AgentRole.RESEARCHER,
                task_type="research",
                priority=workflow_context.get("priority", "medium"),
                domain=workflow_context.get("domain", "trading")
            ),
            AgentRole.MASTERMIND: AgentPromptContext(
                agent_role=AgentRole.MASTERMIND,
                task_type="strategy_design",
                priority=workflow_context.get("priority", "medium"),
                domain=workflow_context.get("domain", "trading")
            ),
            AgentRole.EXECUTOR: AgentPromptContext(
                agent_role=AgentRole.EXECUTOR,
                task_type="implementation",
                priority=workflow_context.get("priority", "medium"),
                domain=workflow_context.get("domain", "trading")
            )
        }
        
        # Compress for each agent in parallel
        tasks = [
            self.compress_agent_prompt(
                self._adapt_prompt_for_agent(workflow_prompt, role, workflow_context),
                context
            )
            for role, context in contexts.items()
        ]
        
        agent_results = await asyncio.gather(*tasks)
        
        # Map results to agent roles
        for role, result in zip(contexts.keys(), agent_results):
            results[role] = result
        
        return results
    
    def _select_compression_config(
        self,
        profile: AgentCompressionProfile,
        context: AgentPromptContext,
        strategy: CompressionStrategy
    ) -> CompressionConfig:
        """Select appropriate compression config for agent and context"""
        
        # Try to get task-specific config
        config = profile.compression_configs.get(context.task_type)
        
        if not config:
            # Fall back to default based on strategy
            if strategy == CompressionStrategy.AGGRESSIVE:
                config = CompressionConfig(compression_ratio=0.3)
            elif strategy == CompressionStrategy.CONSERVATIVE:
                config = CompressionConfig(compression_ratio=0.8)
            elif strategy == CompressionStrategy.BALANCED:
                config = CompressionConfig(compression_ratio=0.6)
            else:
                # Agent-optimized or adaptive
                config = CompressionConfig(compression_ratio=0.5)
        
        # Apply context-specific modifications
        if context.priority == "critical":
            config.compression_ratio = min(config.compression_ratio * 1.5, 0.9)
        elif context.priority == "low":
            config.compression_ratio = max(config.compression_ratio * 0.7, 0.2)
        
        # Respect maximum compression ratio
        config.compression_ratio = min(
            config.compression_ratio, 
            context.max_compression_ratio
        )
        
        return config
    
    async def _post_process_agent_result(
        self,
        result: CompressionResult,
        context: AgentPromptContext,
        profile: AgentCompressionProfile
    ) -> CompressionResult:
        """Post-process compression result for agent-specific requirements"""
        
        # Check if compression ratio is acceptable
        if result.compression_ratio < context.max_compression_ratio:
            logger.warning(
                f"Compression ratio {result.compression_ratio:.2f} exceeds maximum "
                f"{context.max_compression_ratio:.2f} for {context.agent_role.value}"
            )
            
            # If too aggressive, fall back to original
            if context.max_compression_ratio < 0.5:
                result.compressed_prompt = result.original_prompt
                result.compressed_tokens = result.original_tokens
                result.compression_ratio = 1.0
                result.fallback_used = True
        
        # Agent-specific post-processing
        if context.agent_role == AgentRole.RESEARCHER and context.researcher_preserve_sources:
            result.compressed_prompt = self._ensure_sources_preserved(result.compressed_prompt)
        
        elif context.agent_role == AgentRole.MASTERMIND and context.mastermind_preserve_strategy:
            result.compressed_prompt = self._ensure_strategy_preserved(result.compressed_prompt)
        
        elif context.agent_role == AgentRole.EXECUTOR and context.executor_preserve_code:
            result.compressed_prompt = self._ensure_code_preserved(result.compressed_prompt)
        
        return result
    
    def _create_handoff_prompt(
        self,
        from_agent: AgentRole,
        to_agent: AgentRole,
        handoff_data: Dict[str, Any]
    ) -> str:
        """Create formatted handoff prompt"""
        handoff_prompt = f"""
AGENT HANDOFF: {from_agent.value} -> {to_agent.value}

Handoff Data:
{json.dumps(handoff_data, indent=2)}

Context Preservation Required:
- Maintain agent context and state
- Preserve critical decision points
- Transfer actionable insights
- Ensure continuity of workflow

Agent Transition Instructions:
{to_agent.value} should continue the workflow using the provided context and data.
"""
        return handoff_prompt.strip()
    
    def _add_handoff_metadata(
        self,
        compressed_prompt: str,
        from_agent: AgentRole,
        to_agent: AgentRole
    ) -> str:
        """Add handoff metadata to compressed prompt"""
        metadata = f"[HANDOFF: {from_agent.value} -> {to_agent.value}]\n"
        return metadata + compressed_prompt
    
    def _adapt_prompt_for_agent(
        self,
        base_prompt: str,
        agent_role: AgentRole,
        context: Dict[str, Any]
    ) -> str:
        """Adapt base prompt for specific agent role"""
        role_adaptations = {
            AgentRole.RESEARCHER: f"RESEARCHER AGENT TASK:\n{base_prompt}",
            AgentRole.MASTERMIND: f"MASTERMIND AGENT TASK:\n{base_prompt}",
            AgentRole.EXECUTOR: f"EXECUTOR AGENT TASK:\n{base_prompt}"
        }
        
        return role_adaptations.get(agent_role, base_prompt)
    
    def _ensure_sources_preserved(self, prompt: str) -> str:
        """Ensure research sources are preserved in compressed prompt"""
        # Implementation would check for and preserve source references
        return prompt
    
    def _ensure_strategy_preserved(self, prompt: str) -> str:
        """Ensure strategic context is preserved in compressed prompt"""
        # Implementation would check for and preserve strategic elements
        return prompt
    
    def _ensure_code_preserved(self, prompt: str) -> str:
        """Ensure code examples/references are preserved in compressed prompt"""
        # Implementation would check for and preserve code blocks
        return prompt
    
    async def _update_agent_stats(
        self,
        agent_role: AgentRole,
        result: CompressionResult
    ):
        """Update statistics for specific agent"""
        stats = self.agent_stats[agent_role.value]
        
        stats["total_compressions"] += 1
        stats["total_tokens_saved"] += result.token_savings
        stats["cost_savings"] += result.cost_savings_estimate
        
        # Update averages
        count = stats["total_compressions"]
        current_ratio_avg = stats["average_compression_ratio"]
        new_ratio_avg = ((current_ratio_avg * (count - 1)) + result.compression_ratio) / count
        stats["average_compression_ratio"] = new_ratio_avg
        
        if result.quality_score:
            current_quality_avg = stats["quality_score"]
            new_quality_avg = ((current_quality_avg * (count - 1)) + result.quality_score) / count
            stats["quality_score"] = new_quality_avg
    
    async def _log_agent_compression(
        self,
        context: AgentPromptContext,
        result: CompressionResult
    ):
        """Log agent-specific compression metrics"""
        try:
            # Store in compression history for analysis
            self.compression_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "agent_role": context.agent_role.value,
                "task_type": context.task_type,
                "priority": context.priority,
                "domain": context.domain,
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "compression_ratio": result.compression_ratio,
                "processing_time_ms": result.processing_time_ms,
                "model_used": result.model_used,
                "fallback_used": result.fallback_used,
                "cost_savings": result.cost_savings_estimate
            })
            
            # Keep only recent history (last 1000 entries)
            if len(self.compression_history) > 1000:
                self.compression_history = self.compression_history[-1000:]
                
        except Exception as e:
            logger.warning(f"Failed to log agent compression: {e}")
    
    async def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent compression statistics"""
        return {
            "agent_stats": self.agent_stats,
            "total_history_entries": len(self.compression_history),
            "agent_profiles": {
                role.value: {
                    "default_strategy": profile.default_strategy.value,
                    "preservation_rules": profile.preservation_rules,
                    "cost_optimization_priority": profile.cost_optimization_priority
                }
                for role, profile in self.agent_profiles.items()
            }
        }
    
    async def optimize_agent_profiles(self):
        """Optimize agent compression profiles based on historical performance"""
        # Analyze compression history to optimize profiles
        # This would implement machine learning-based optimization
        logger.info("Agent profile optimization not yet implemented")
        pass


# Global agent compression service
agent_compression_service = None

async def get_agent_compression_service(
    llmlingua_service: LLMLinguaService
) -> AgentCompressionService:
    """Get or create agent compression service"""
    global agent_compression_service
    if agent_compression_service is None:
        agent_compression_service = AgentCompressionService(llmlingua_service)
    return agent_compression_service