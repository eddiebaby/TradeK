"""
Agent Router - SPARC Trio Integration with TradeKnowledge

This module provides API endpoints for interacting with the SPARC trio agent system,
integrating MASTERMIND, EXECUTOR, and RESEARCHER agents with the existing
TradeKnowledge infrastructure.
"""

import logging

# Agent system imports
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from src.ingestion.enhanced_book_processor import EnhancedBookProcessor

# TradeKnowledge core imports
from src.search.unified_search import UnifiedSearchEngine
from src.utils.cache_manager import CacheManager

sys.path.append(str(Path(__file__).parent.parent.parent / "agents"))

from executor.executor_agent import ExecutorAgent
from mastermind.mastermind_agent import MastermindAgent
from researcher.researcher_agent import ResearcherAgent
from sparc.sparc_orchestrator import SPARCOrchestrator, SPARCPhase

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for API requests/responses


class ProjectSpec(BaseModel):
    """Project specification for SPARC workflow initiation."""

    title: str = Field(..., description="Project title")
    description: str = Field(..., description="Project description")
    requirements: dict[str, Any] = Field(
        default_factory=dict, description="Project requirements"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Project constraints"
    )
    priority: str = Field(default="medium", description="Project priority")
    target_completion: str | None = Field(
        None, description="Target completion timeline"
    )


class ResearchRequest(BaseModel):
    """Research request for RESEARCHER agent."""

    topic: str = Field(..., description="Research topic")
    domains: list[str] = Field(default_factory=list, description="Research domains")
    depth: str = Field(default="standard", description="Research depth")
    focus_areas: list[str] = Field(default_factory=list, description="Focus areas")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )


class ImplementationRequest(BaseModel):
    """Implementation request for EXECUTOR agent."""

    task_description: str = Field(..., description="Implementation task description")
    requirements: dict[str, Any] = Field(
        default_factory=dict, description="Implementation requirements"
    )
    quality_targets: dict[str, Any] = Field(
        default_factory=dict, description="Quality targets"
    )
    performance_targets: dict[str, Any] = Field(
        default_factory=dict, description="Performance targets"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Implementation constraints"
    )


class StrategicAnalysisRequest(BaseModel):
    """Strategic analysis request for MASTERMIND agent."""

    project_context: dict[str, Any] = Field(..., description="Project context")
    analysis_scope: list[str] = Field(
        default_factory=list, description="Analysis scope"
    )
    decision_points: list[str] = Field(
        default_factory=list, description="Key decision points"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Strategic constraints"
    )


# Global agent instances (will be initialized in lifespan)
sparc_orchestrator: SPARCOrchestrator | None = None
mastermind_agent: MastermindAgent | None = None
executor_agent: ExecutorAgent | None = None
researcher_agent: ResearcherAgent | None = None


# Dependency injection functions
async def get_search_engine() -> UnifiedSearchEngine:
    """Get the unified search engine instance."""
    # This will be injected from the main app state
    from src.api.main import app_state

    return app_state.get("search_engine")


async def get_book_processor() -> EnhancedBookProcessor:
    """Get the enhanced book processor instance."""
    from src.api.main import app_state

    return app_state.get("book_processor")


async def get_cache_manager() -> CacheManager:
    """Get the cache manager instance."""
    from src.api.main import app_state

    return app_state.get("cache_manager")


# Agent initialization
async def initialize_agents(
    search_engine: UnifiedSearchEngine,
    book_processor: EnhancedBookProcessor,
    cache_manager: CacheManager,
):
    """Initialize the SPARC trio agent system with TradeKnowledge integrations."""
    global sparc_orchestrator, mastermind_agent, executor_agent, researcher_agent

    logger.info("Initializing SPARC trio agent system...")

    # Initialize individual agents
    mastermind_agent = MastermindAgent()
    executor_agent = ExecutorAgent()
    researcher_agent = ResearcherAgent()

    # Enhance RESEARCHER agent with TradeKnowledge search capabilities
    researcher_agent.search_engine = search_engine
    researcher_agent.book_processor = book_processor

    # Enhance EXECUTOR agent with TradeKnowledge processing capabilities
    executor_agent.book_processor = book_processor
    executor_agent.cache_manager = cache_manager

    # Initialize SPARC orchestrator with agents
    sparc_orchestrator = SPARCOrchestrator(
        mastermind_agent=mastermind_agent,
        executor_agent=executor_agent,
        researcher_agent=researcher_agent,
    )

    # Add TradeKnowledge service integrations
    sparc_orchestrator.search_engine = search_engine
    sparc_orchestrator.book_processor = book_processor
    sparc_orchestrator.cache_manager = cache_manager

    logger.info("SPARC trio agent system initialized successfully")


# API Endpoints


@router.get("/health")
async def agent_health_check():
    """Health check for agent system."""
    if not sparc_orchestrator:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    return {
        "status": "healthy",
        "agents": {
            "mastermind": "active" if mastermind_agent else "inactive",
            "executor": "active" if executor_agent else "inactive",
            "researcher": "active" if researcher_agent else "inactive",
            "sparc_orchestrator": "active" if sparc_orchestrator else "inactive",
        },
        "active_projects": (
            len(sparc_orchestrator.get_active_projects()) if sparc_orchestrator else 0
        ),
        "timestamp": time.time(),
    }


@router.post("/sparc/projects")
async def create_sparc_project(
    project_spec: ProjectSpec, background_tasks: BackgroundTasks
):
    """Create a new SPARC project and initiate the workflow."""
    if not sparc_orchestrator:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    try:
        # Convert ProjectSpec to SPARC project specification
        sparc_spec = {
            "title": project_spec.title,
            "description": project_spec.description,
            "requirements": project_spec.requirements,
            "constraints": project_spec.constraints,
            "priority": project_spec.priority,
            "target_completion": project_spec.target_completion,
        }

        # Initiate SPARC project
        project = await sparc_orchestrator.initiate_sparc_project(sparc_spec)

        # Start workflow execution in background
        background_tasks.add_task(execute_sparc_workflow_background, project.project_id)

        return {
            "project_id": project.project_id,
            "title": project.title,
            "status": "initiated",
            "current_phase": project.current_phase.value,
            "message": "SPARC project initiated and workflow started",
            "tracking_url": f"/api/v1/agents/sparc/projects/{project.project_id}/status",
        }

    except Exception as e:
        logger.error(f"Failed to create SPARC project: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create project: {str(e)}"
        )


@router.get("/sparc/projects/{project_id}/status")
async def get_project_status(project_id: str):
    """Get the status of a SPARC project."""
    if not sparc_orchestrator:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    active_projects = sparc_orchestrator.get_active_projects()
    project = active_projects.get(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return {
        "project_id": project.project_id,
        "title": project.title,
        "description": project.description,
        "current_phase": project.current_phase.value,
        "phase_history": project.phase_history,
        "quality_metrics": project.quality_metrics,
        "deliverables": {
            phase.value: len(deliverables)
            for phase, deliverables in project.deliverables.items()
        },
        "created_at": project.created_at,
        "updated_at": project.updated_at,
        "duration": time.time() - project.created_at,
    }


@router.post("/sparc/projects/{project_id}/execute")
async def execute_sparc_project(project_id: str, background_tasks: BackgroundTasks):
    """Execute or resume SPARC workflow for a project."""
    if not sparc_orchestrator:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    active_projects = sparc_orchestrator.get_active_projects()
    if project_id not in active_projects:
        raise HTTPException(status_code=404, detail="Project not found")

    # Start workflow execution in background
    background_tasks.add_task(execute_sparc_workflow_background, project_id)

    return {
        "project_id": project_id,
        "status": "execution_started",
        "message": "SPARC workflow execution initiated",
        "tracking_url": f"/api/v1/agents/sparc/projects/{project_id}/status",
    }


@router.post("/research")
async def request_research(
    request: ResearchRequest,
    search_engine: UnifiedSearchEngine = Depends(get_search_engine),
):
    """Request research from the RESEARCHER agent."""
    if not researcher_agent:
        raise HTTPException(status_code=503, detail="RESEARCHER agent not initialized")

    try:
        # Enhance research with TradeKnowledge search capabilities
        research_spec = {
            "topic": request.topic,
            "domains": request.domains or ["technical_deep_dive"],
            "depth": request.depth,
            "focus_areas": request.focus_areas,
            "context": request.context,
            "search_integration": True,
        }

        # Conduct research with search engine integration
        if (
            hasattr(researcher_agent, "search_engine")
            and researcher_agent.search_engine
        ):
            # Use integrated search for knowledge base research
            search_results = await researcher_agent.search_engine.search(request.topic)
            research_spec["context"]["knowledge_base_results"] = search_results

        # Execute research
        research_results = await researcher_agent.conduct_comprehensive_research(
            research_spec
        )

        return {
            "research_id": research_results.research_id,
            "topic": request.topic,
            "domains_covered": [
                d.value for d in research_results.request.research_domains
            ],
            "insights_count": len(research_results.insights),
            "quality_metrics": research_results.quality_metrics,
            "summary": research_results.summary,
            "research_duration": research_results.research_duration,
            "actionable_insights": [
                {
                    "title": insight.title,
                    "recommendations": insight.actionable_recommendations[:3],
                    "confidence": insight.confidence_score,
                }
                for insight in research_results.insights[:5]
            ],
        }

    except Exception as e:
        logger.error(f"Research request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


@router.post("/implementation")
async def request_implementation(request: ImplementationRequest):
    """Request implementation from the EXECUTOR agent."""
    if not executor_agent:
        raise HTTPException(status_code=503, detail="EXECUTOR agent not initialized")

    try:
        # Create task context for EXECUTOR
        from core.agent_base import TaskContext

        task_context = TaskContext(
            task_id=f"impl_{int(time.time() * 1000)}",
            description=request.task_description,
            requirements=request.requirements,
            constraints=request.constraints,
            performance_targets=request.performance_targets,
            priority="medium",
            start_time=time.time(),
        )

        # Execute implementation
        implementation_result = await executor_agent.process_task(task_context)

        return {
            "task_id": task_context.task_id,
            "status": "completed",
            "implementation_summary": implementation_result.get(
                "execution_summary", {}
            ),
            "quality_metrics": implementation_result.get("quality_report", {}),
            "test_coverage": implementation_result.get("test_suite", {}).get(
                "coverage_report", {}
            ),
            "performance_metrics": implementation_result.get(
                "implementation_result", {}
            ).get("performance_metrics", {}),
            "security_metrics": implementation_result.get(
                "implementation_result", {}
            ).get("security_metrics", {}),
            "recommendations": implementation_result.get("execution_summary", {}).get(
                "next_steps", []
            ),
        }

    except Exception as e:
        logger.error(f"Implementation request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Implementation failed: {str(e)}")


@router.post("/strategy")
async def request_strategic_analysis(request: StrategicAnalysisRequest):
    """Request strategic analysis from the MASTERMIND agent."""
    if not mastermind_agent:
        raise HTTPException(status_code=503, detail="MASTERMIND agent not initialized")

    try:
        # Create task context for MASTERMIND
        from core.agent_base import TaskContext

        task_context = TaskContext(
            task_id=f"strategy_{int(time.time() * 1000)}",
            description=f"Strategic analysis: {', '.join(request.analysis_scope)}",
            requirements=request.project_context,
            constraints=request.constraints,
            priority="high",
            start_time=time.time(),
        )

        # Execute strategic analysis
        analysis_result = await mastermind_agent.process_task(task_context)

        return {
            "analysis_id": task_context.task_id,
            "strategic_insights": analysis_result.get("strategic_analysis", {}),
            "architecture_recommendations": analysis_result.get(
                "architecture_design", {}
            ),
            "quality_strategy": analysis_result.get("quality_strategy", {}),
            "risk_assessment": analysis_result.get("risk_assessment", {}),
            "executor_recommendations": analysis_result.get(
                "executor_recommendations", {}
            ),
            "decision_confidence": mastermind_agent.strategic_accuracy_score,
            "recommendations": [
                "Implement recommended architecture patterns",
                "Follow quality strategy guidelines",
                "Monitor identified risk factors",
                "Execute with suggested approach",
            ],
        }

    except Exception as e:
        logger.error(f"Strategic analysis request failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Strategic analysis failed: {str(e)}"
        )


@router.get("/metrics")
async def get_agent_metrics():
    """Get performance metrics for the agent system."""
    if not sparc_orchestrator:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    metrics = sparc_orchestrator.get_orchestration_metrics()
    active_projects = sparc_orchestrator.get_active_projects()

    # Agent-specific metrics
    agent_metrics = {}

    if mastermind_agent:
        agent_metrics["mastermind"] = {
            "strategic_accuracy_score": mastermind_agent.strategic_accuracy_score,
            "decision_confidence_threshold": mastermind_agent.decision_confidence_threshold,
            "capabilities": mastermind_agent.get_capabilities(),
        }

    if executor_agent:
        agent_metrics["executor"] = {
            "implementation_quality_score": executor_agent.implementation_quality_score,
            "test_creation_efficiency": executor_agent.test_creation_efficiency,
            "deployment_success_rate": executor_agent.deployment_success_rate,
            "capabilities": executor_agent.get_capabilities(),
        }

    if researcher_agent:
        agent_metrics["researcher"] = {
            "research_accuracy_score": researcher_agent.research_accuracy_score,
            "insight_relevance_threshold": researcher_agent.insight_relevance_threshold,
            "research_confidence_threshold": researcher_agent.research_confidence_threshold,
            "capabilities": researcher_agent.get_capabilities(),
        }

    return {
        "orchestration_metrics": metrics,
        "agent_metrics": agent_metrics,
        "active_projects_count": len(active_projects),
        "active_projects": list(active_projects.keys()),
        "system_status": "operational",
        "timestamp": time.time(),
    }


@router.get("/capabilities")
async def get_agent_capabilities():
    """Get comprehensive capabilities of all agents."""
    capabilities = {
        "sparc_orchestrator": {
            "phases": [phase.value for phase in SPARCPhase],
            "workflow_management": True,
            "quality_gates": True,
            "agent_coordination": True,
        }
    }

    if mastermind_agent:
        capabilities["mastermind"] = {
            "capabilities": mastermind_agent.get_capabilities(),
            "thinking_modes": mastermind_agent.get_thinking_modes(),
            "specializations": [
                "Strategic architecture design",
                "Quality strategy orchestration",
                "Risk assessment and prediction",
                "Technical decision making",
            ],
        }

    if executor_agent:
        capabilities["executor"] = {
            "capabilities": executor_agent.get_capabilities(),
            "execution_modes": executor_agent.get_thinking_modes(),
            "specializations": [
                "TDD implementation",
                "Comprehensive testing",
                "DevOps automation",
                "Performance optimization",
            ],
        }

    if researcher_agent:
        capabilities["researcher"] = {
            "capabilities": researcher_agent.get_capabilities(),
            "research_modes": researcher_agent.get_research_modes(),
            "specializations": [
                "Multi-source intelligence gathering",
                "Evidence-based insight synthesis",
                "Trend analysis and prediction",
                "Best practice identification",
            ],
        }

    return capabilities


# Background task functions


async def execute_sparc_workflow_background(project_id: str):
    """Execute SPARC workflow in background."""
    try:
        logger.info(f"Starting SPARC workflow execution for project {project_id}")
        result = await sparc_orchestrator.execute_sparc_workflow(project_id)
        logger.info(f"SPARC workflow completed for project {project_id}")

        # Cache results for later retrieval
        if sparc_orchestrator.cache_manager:
            cache_key = f"sparc_results:{project_id}"
            await sparc_orchestrator.cache_manager.set(cache_key, result, expire=3600)

    except Exception as e:
        logger.error(f"SPARC workflow execution failed for project {project_id}: {e}")


# Integration with TradeKnowledge search capabilities


@router.post("/research/knowledge-base")
async def research_knowledge_base(
    query: str,
    max_results: int = 10,
    search_engine: UnifiedSearchEngine = Depends(get_search_engine),
):
    """Research using TradeKnowledge knowledge base."""
    if not researcher_agent or not search_engine:
        raise HTTPException(status_code=503, detail="Research system not available")

    try:
        # Use TradeKnowledge search engine for research
        search_results = await search_engine.search(query, max_results=max_results)

        # Enhance results with RESEARCHER analysis
        research_spec = {
            "topic": query,
            "domains": ["technical_deep_dive", "best_practices"],
            "depth": "standard",
            "context": {"search_results": search_results},
        }

        research_intelligence = await researcher_agent.conduct_comprehensive_research(
            research_spec
        )

        return {
            "query": query,
            "search_results_count": len(search_results.get("results", [])),
            "research_insights": [
                {
                    "title": insight.title,
                    "description": insight.description,
                    "recommendations": insight.actionable_recommendations,
                    "confidence": insight.confidence_score,
                }
                for insight in research_intelligence.insights
            ],
            "summary": research_intelligence.summary,
            "benchmarks": research_intelligence.benchmarks,
            "best_practices": research_intelligence.best_practices,
        }

    except Exception as e:
        logger.error(f"Knowledge base research failed: {e}")
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")
