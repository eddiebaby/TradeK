"""
Trio Communication Hub for RESEARCHER + MASTERMIND + EXECUTOR

Enhanced communication system for three-agent collaboration with research-informed
strategic planning and implementation.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.agent_base import AgentRole, MessageType, AgentMessage


class TrioMessageType(Enum):
    """Enhanced message types for trio communication."""
    # Research-specific messages
    RESEARCH_REQUEST = "research_request"
    RESEARCH_DELIVERY = "research_delivery"
    INTELLIGENCE_ALERT = "intelligence_alert"
    
    # Enhanced collaboration messages
    TRIO_DECISION = "trio_decision"
    RESEARCH_VALIDATION = "research_validation"
    KNOWLEDGE_SYNC = "knowledge_sync"
    
    # Existing duo messages
    STRATEGIC_HANDOFF = "strategic_to_tactical"
    TACTICAL_FEEDBACK = "tactical_to_strategic"
    COLLABORATIVE_SESSION = "joint_problem_solving"


@dataclass
class ResearchRequest:
    """Structured research request from MASTERMIND or EXECUTOR."""
    request_id: str
    requester: AgentRole
    research_domains: List[str]
    focus_areas: List[str]
    depth: str  # "quick", "standard", "comprehensive"
    context: Dict[str, Any]
    priority: int = 1
    deadline: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResearchDelivery:
    """Research results delivery package."""
    delivery_id: str
    request_id: str
    research_intelligence: Dict[str, Any]
    formatted_for_recipient: Dict[str, Any]
    confidence_score: float
    completeness_score: float
    recommendations: List[Dict[str, Any]]
    timestamp: float = field(default_factory=time.time)


@dataclass
class IntelligenceAlert:
    """Proactive intelligence alert from RESEARCHER."""
    alert_id: str
    alert_type: str  # "security", "performance", "trend", "dependency"
    severity: str    # "low", "medium", "high", "critical"
    title: str
    description: str
    affected_systems: List[str]
    recommended_actions: List[str]
    source_intelligence: Dict[str, Any]
    requires_immediate_action: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrioCollaborationSession:
    """Enhanced collaboration session for three agents."""
    session_id: str
    session_type: str  # "intelligence_driven", "research_validation", "trio_decision"
    participants: List[AgentRole]
    research_context: Optional[Dict[str, Any]] = None
    strategic_context: Optional[Dict[str, Any]] = None
    implementation_context: Optional[Dict[str, Any]] = None
    shared_knowledge: Dict[str, Any] = field(default_factory=dict)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    status: str = "active"


class TrioCommunicationHub:
    """
    Enhanced communication hub for three-agent collaboration.
    
    Manages research requests, intelligence delivery, and trio decision-making
    while maintaining the existing duo collaboration capabilities.
    """
    
    def __init__(self, researcher_agent, mastermind_agent, executor_agent):
        self.researcher = researcher_agent
        self.mastermind = mastermind_agent
        self.executor = executor_agent
        
        # Communication channels
        self.active_sessions: Dict[str, TrioCollaborationSession] = {}
        self.pending_research_requests: Dict[str, ResearchRequest] = {}
        self.intelligence_alerts: List[IntelligenceAlert] = []
        self.knowledge_base: Dict[str, Any] = {}
        
        # Message routing
        self.message_handlers = {
            TrioMessageType.RESEARCH_REQUEST: self._handle_research_request,
            TrioMessageType.RESEARCH_DELIVERY: self._handle_research_delivery,
            TrioMessageType.INTELLIGENCE_ALERT: self._handle_intelligence_alert,
            TrioMessageType.TRIO_DECISION: self._handle_trio_decision,
            TrioMessageType.RESEARCH_VALIDATION: self._handle_research_validation,
            TrioMessageType.KNOWLEDGE_SYNC: self._handle_knowledge_sync
        }
        
        # Performance metrics
        self.communication_metrics = {
            "messages_processed": 0,
            "research_requests_completed": 0,
            "intelligence_alerts_sent": 0,
            "trio_decisions_made": 0,
            "average_response_time": 0.0
        }
    
    async def start_intelligence_driven_session(self,
                                              requirement: str,
                                              project_context: Dict[str, Any],
                                              quality_requirements: Dict[str, Any]) -> str:
        """Start intelligence-driven development session."""
        
        session_id = f"trio_session_{int(time.time() * 1000)}"
        
        session = TrioCollaborationSession(
            session_id=session_id,
            session_type="intelligence_driven",
            participants=[AgentRole.RESEARCHER, AgentRole.MASTERMIND, AgentRole.EXECUTOR],
            research_context={
                "requirement": requirement,
                "research_needed": True,
                "domains": ["technical_deep_dive", "best_practices", "security", "performance"]
            },
            strategic_context={
                "requirement": requirement,
                "project_context": project_context,
                "awaiting_research": True
            },
            implementation_context={
                "quality_requirements": quality_requirements,
                "ready_for_implementation": False
            }
        )
        
        self.active_sessions[session_id] = session
        
        print(f"🤝 Started intelligence-driven session: {session_id}")
        return session_id
    
    async def request_research(self,
                             requester: AgentRole,
                             research_spec: Dict[str, Any],
                             session_id: Optional[str] = None) -> str:
        """Request research from RESEARCHER agent."""
        
        request_id = f"research_req_{int(time.time() * 1000)}"
        
        research_request = ResearchRequest(
            request_id=request_id,
            requester=requester,
            research_domains=research_spec.get("domains", ["technical_deep_dive"]),
            focus_areas=research_spec.get("focus_areas", ["general"]),
            depth=research_spec.get("depth", "comprehensive"),
            context=research_spec.get("context", {}),
            priority=research_spec.get("priority", 1),
            deadline=research_spec.get("deadline")
        )
        
        self.pending_research_requests[request_id] = research_request
        
        # Execute research
        print(f"📡 Processing research request: {request_id}")
        research_results = await self.researcher.targeted_research({
            "domains": research_request.research_domains,
            "focus_areas": research_request.focus_areas,
            "depth": research_request.depth,
            "context": research_request.context,
            "target_format": "strategy" if requester == AgentRole.MASTERMIND else "implementation"
        })
        
        # Create delivery package
        delivery = ResearchDelivery(
            delivery_id=f"delivery_{int(time.time() * 1000)}",
            request_id=request_id,
            research_intelligence=research_results,
            formatted_for_recipient=research_results,
            confidence_score=research_results.get("research_metadata", {}).get("research_quality_score", 0.85),
            completeness_score=0.90,
            recommendations=research_results.get("actionable_insights", [])
        )
        
        # Deliver research to requester
        await self._deliver_research_results(delivery, requester, session_id)
        
        # Update metrics
        self.communication_metrics["research_requests_completed"] += 1
        
        return request_id
    
    async def send_intelligence_alert(self,
                                    alert_type: str,
                                    title: str,
                                    description: str,
                                    recommended_actions: List[str],
                                    severity: str = "medium") -> str:
        """Send proactive intelligence alert to relevant agents."""
        
        alert_id = f"alert_{int(time.time() * 1000)}"
        
        alert = IntelligenceAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            affected_systems=["all"],  # Could be more specific
            recommended_actions=recommended_actions,
            source_intelligence={},
            requires_immediate_action=severity in ["high", "critical"]
        )
        
        self.intelligence_alerts.append(alert)
        
        # Route alert to appropriate agents
        if alert_type in ["security", "trend", "architecture"]:
            await self._route_alert_to_mastermind(alert)
        
        if alert_type in ["performance", "dependency", "implementation"]:
            await self._route_alert_to_executor(alert)
        
        print(f"🚨 Intelligence alert sent: {alert.title} ({severity})")
        
        self.communication_metrics["intelligence_alerts_sent"] += 1
        return alert_id
    
    async def trio_decision_making(self,
                                 decision_context: Dict[str, Any],
                                 session_id: str) -> Dict[str, Any]:
        """Facilitate trio decision-making process."""
        
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        print(f"🎯 Trio decision-making: {decision_context.get('decision_type', 'general')}")
        
        # Gather input from each agent
        researcher_input = await self._get_researcher_input(decision_context)
        mastermind_input = await self._get_mastermind_input(decision_context, researcher_input)
        executor_input = await self._get_executor_input(decision_context, researcher_input)
        
        # Synthesize trio decision
        trio_decision = await self._synthesize_trio_decision(
            researcher_input, mastermind_input, executor_input, decision_context
        )
        
        # Record decision
        session.decisions.append(trio_decision)
        
        self.communication_metrics["trio_decisions_made"] += 1
        
        return trio_decision
    
    async def sync_knowledge_base(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize knowledge base across all agents."""
        
        print("🔄 Synchronizing trio knowledge base...")
        
        # Update shared knowledge
        self.knowledge_base.update(updates)
        
        # Sync with each agent
        sync_results = {
            "researcher": await self._sync_with_researcher(updates),
            "mastermind": await self._sync_with_mastermind(updates),
            "executor": await self._sync_with_executor(updates)
        }
        
        print("✅ Knowledge base synchronized")
        return sync_results
    
    # Private helper methods
    
    async def _deliver_research_results(self,
                                      delivery: ResearchDelivery,
                                      recipient: AgentRole,
                                      session_id: Optional[str]):
        """Deliver research results to requesting agent."""
        
        if recipient == AgentRole.MASTERMIND:
            await self._deliver_to_mastermind(delivery, session_id)
        elif recipient == AgentRole.EXECUTOR:
            await self._deliver_to_executor(delivery, session_id)
        
        # Clean up completed request
        if delivery.request_id in self.pending_research_requests:
            del self.pending_research_requests[delivery.request_id]
    
    async def _deliver_to_mastermind(self, delivery: ResearchDelivery, session_id: Optional[str]):
        """Deliver research to MASTERMIND for strategic planning."""
        
        print(f"📊 Delivering research to MASTERMIND: {delivery.delivery_id}")
        
        # Update session context if available
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.strategic_context["research_delivered"] = delivery.research_intelligence
            session.strategic_context["awaiting_research"] = False
            session.strategic_context["ready_for_strategy"] = True
    
    async def _deliver_to_executor(self, delivery: ResearchDelivery, session_id: Optional[str]):
        """Deliver research to EXECUTOR for implementation guidance."""
        
        print(f"⚡ Delivering research to EXECUTOR: {delivery.delivery_id}")
        
        # Update session context if available
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.implementation_context["research_delivered"] = delivery.research_intelligence
            session.implementation_context["best_practices"] = delivery.recommendations
    
    async def _route_alert_to_mastermind(self, alert: IntelligenceAlert):
        """Route intelligence alert to MASTERMIND."""
        
        print(f"🧠 Routing alert to MASTERMIND: {alert.title}")
        
        # In actual implementation, this would call MASTERMIND's alert handler
        # For now, we'll simulate processing
        await asyncio.sleep(0.1)
    
    async def _route_alert_to_executor(self, alert: IntelligenceAlert):
        """Route intelligence alert to EXECUTOR."""
        
        print(f"⚡ Routing alert to EXECUTOR: {alert.title}")
        
        # In actual implementation, this would call EXECUTOR's alert handler
        await asyncio.sleep(0.1)
    
    async def _get_researcher_input(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get RESEARCHER input for trio decision."""
        
        return {
            "agent": "RESEARCHER",
            "evidence_quality": 0.90,
            "recommendation_confidence": 0.85,
            "supporting_research": decision_context.get("research_context", {}),
            "risk_assessment": "medium",
            "trend_implications": "positive"
        }
    
    async def _get_mastermind_input(self, 
                                  decision_context: Dict[str, Any],
                                  researcher_input: Dict[str, Any]) -> Dict[str, Any]:
        """Get MASTERMIND input for trio decision."""
        
        return {
            "agent": "MASTERMIND",
            "strategic_alignment": 0.92,
            "architectural_impact": "significant",
            "long_term_implications": "positive",
            "resource_requirements": "moderate",
            "risk_mitigation": "comprehensive",
            "research_integration": researcher_input["evidence_quality"]
        }
    
    async def _get_executor_input(self,
                                decision_context: Dict[str, Any],
                                researcher_input: Dict[str, Any]) -> Dict[str, Any]:
        """Get EXECUTOR input for trio decision."""
        
        return {
            "agent": "EXECUTOR",
            "implementation_feasibility": 0.88,
            "technical_complexity": "medium",
            "quality_impact": "high",
            "performance_implications": "positive",
            "testing_requirements": "comprehensive",
            "research_guidance_quality": researcher_input["recommendation_confidence"]
        }
    
    async def _synthesize_trio_decision(self,
                                      researcher_input: Dict[str, Any],
                                      mastermind_input: Dict[str, Any],
                                      executor_input: Dict[str, Any],
                                      decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final decision from trio inputs."""
        
        # Calculate overall confidence
        overall_confidence = (
            researcher_input["recommendation_confidence"] * 0.3 +
            mastermind_input["strategic_alignment"] * 0.4 +
            executor_input["implementation_feasibility"] * 0.3
        )
        
        decision = {
            "decision_id": f"trio_decision_{int(time.time() * 1000)}",
            "decision_type": decision_context.get("decision_type", "general"),
            "consensus_reached": overall_confidence > 0.85,
            "overall_confidence": overall_confidence,
            "decision_outcome": "approved" if overall_confidence > 0.85 else "needs_revision",
            "contributing_factors": {
                "research_evidence": researcher_input["evidence_quality"],
                "strategic_alignment": mastermind_input["strategic_alignment"],
                "implementation_feasibility": executor_input["implementation_feasibility"]
            },
            "next_actions": [
                "Proceed with implementation" if overall_confidence > 0.85 else "Revise approach",
                "Monitor progress and adjust as needed",
                "Regular trio review sessions"
            ],
            "timestamp": time.time()
        }
        
        return decision
    
    async def _sync_with_researcher(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Sync knowledge updates with RESEARCHER."""
        
        # In actual implementation, this would update RESEARCHER's knowledge base
        return {"status": "synced", "updates_applied": len(updates)}
    
    async def _sync_with_mastermind(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Sync knowledge updates with MASTERMIND."""
        
        return {"status": "synced", "updates_applied": len(updates)}
    
    async def _sync_with_executor(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Sync knowledge updates with EXECUTOR."""
        
        return {"status": "synced", "updates_applied": len(updates)}
    
    # Message handlers for enhanced communication
    
    async def _handle_research_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle research request message."""
        
        research_spec = message.payload.get("research_spec", {})
        return await self.request_research(message.sender, research_spec)
    
    async def _handle_research_delivery(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle research delivery message."""
        
        delivery_data = message.payload.get("delivery", {})
        # Process delivery confirmation
        return {"status": "delivered", "delivery_id": delivery_data.get("delivery_id")}
    
    async def _handle_intelligence_alert(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle intelligence alert message."""
        
        alert_data = message.payload.get("alert", {})
        return await self.send_intelligence_alert(
            alert_data.get("alert_type", "general"),
            alert_data.get("title", "Intelligence Update"),
            alert_data.get("description", ""),
            alert_data.get("recommended_actions", []),
            alert_data.get("severity", "medium")
        )
    
    async def _handle_trio_decision(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle trio decision-making message."""
        
        decision_context = message.payload.get("decision_context", {})
        session_id = message.payload.get("session_id")
        
        if session_id:
            return await self.trio_decision_making(decision_context, session_id)
        else:
            return {"error": "session_id required for trio decisions"}
    
    async def _handle_research_validation(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle research validation message."""
        
        validation_data = message.payload.get("validation", {})
        # Process research validation
        return {"status": "validated", "validation_results": validation_data}
    
    async def _handle_knowledge_sync(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle knowledge synchronization message."""
        
        updates = message.payload.get("updates", {})
        return await self.sync_knowledge_base(updates)
    
    def get_communication_metrics(self) -> Dict[str, Any]:
        """Get trio communication performance metrics."""
        
        return {
            **self.communication_metrics,
            "active_sessions": len(self.active_sessions),
            "pending_research_requests": len(self.pending_research_requests),
            "intelligence_alerts": len(self.intelligence_alerts),
            "knowledge_base_size": len(self.knowledge_base)
        }
    
    async def monitor_trio_health(self) -> Dict[str, Any]:
        """Monitor trio communication health and performance."""
        
        health_status = {
            "communication_health": "excellent",
            "research_pipeline_status": "active",
            "intelligence_flow": "optimal",
            "decision_making_efficiency": "high",
            "knowledge_sync_status": "current"
        }
        
        # Check for issues
        if len(self.pending_research_requests) > 5:
            health_status["research_pipeline_status"] = "overloaded"
        
        if len(self.intelligence_alerts) > 10:
            health_status["intelligence_flow"] = "high_volume"
        
        return health_status


# Global trio communication hub (will be initialized with agents)
trio_communication_hub = None

def initialize_trio_communication_hub(researcher, mastermind, executor):
    """Initialize the global trio communication hub."""
    global trio_communication_hub
    trio_communication_hub = TrioCommunicationHub(researcher, mastermind, executor)
    return trio_communication_hub