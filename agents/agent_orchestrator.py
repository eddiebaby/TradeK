"""
Agent Orchestration System for MASTERMIND & EXECUTOR Coordination

This module orchestrates the collaboration between MASTERMIND and EXECUTOR agents,
managing task handoffs, communication, and ensuring optimal performance.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from core.agent_base import BaseAgent, AgentRole, TaskContext, MessageType, communication_bus
from mastermind.mastermind_agent import MastermindAgent
from executor.executor_agent import ExecutorAgent
from tools.mcp_communication import communication_hub, HandoffPackage
from tools.mcp_codebase_analyzer import codebase_analyzer
from tools.mcp_code_generator import code_generator
from tools.mcp_testing_framework import testing_framework
from tools.mcp_devops_automation import devops_automation
from tools.mcp_monitoring_metrics import monitoring_metrics
from collaboration_patterns import initialize_collaboration_patterns, strategic_implementation_cycle, continuous_quality_amplification
from intelligence_amplification import intelligence_engine


class AgentOrchestrator:
    """
    Orchestrates collaboration between MASTERMIND and EXECUTOR agents.
    
    Manages task flow, communication, quality assurance, and performance optimization
    to achieve the highest possible code quality and implementation excellence.
    """
    
    def __init__(self):
        # Initialize agents
        self.mastermind = MastermindAgent()
        self.executor = ExecutorAgent()
        
        # Register agents with communication bus
        communication_bus.register_agent(self.mastermind)
        communication_bus.register_agent(self.executor)
        
        # Initialize enhanced collaboration patterns
        initialize_collaboration_patterns(self.mastermind, self.executor)
        
        # Register MCP tools with agents
        self._register_mcp_tools()
        
        # Orchestration state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Enhanced quality thresholds based on plan
        self.quality_thresholds = {
            "test_coverage": 95,  # Increased from 90
            "mutation_score": 85,  # Increased from 80
            "response_time": 100,  # ms
            "security_score": 9.5,  # Increased from 9.0
            "maintainability": 9.0,  # Increased from 8.0
            "strategic_accuracy": 0.88,
            "collaboration_effectiveness": 0.85,
            "quality_amplification": 1.2
        }
    
    async def execute_comprehensive_development_cycle(self,
                                                    requirement: str,
                                                    project_context: Dict[str, Any],
                                                    quality_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complete development cycle with MASTERMIND strategic analysis
        followed by EXECUTOR implementation using TDD principles.
        
        Args:
            requirement: Development requirement or user story
            project_context: Project context and constraints
            quality_requirements: Quality standards and requirements
            
        Returns:
            Dict[str, Any]: Complete development cycle results
        """
        cycle_start = time.time()
        session_id = f"dev_cycle_{int(time.time() * 1000)}"
        
        # Initialize development session
        session = {
            "session_id": session_id,
            "requirement": requirement,
            "project_context": project_context,
            "quality_requirements": quality_requirements,
            "start_time": cycle_start,
            "phases": [],
            "status": "in_progress",
            "final_results": {}
        }
        
        self.active_sessions[session_id] = session
        
        try:
            # Enhanced Strategic Implementation Cycle
            print("🚀 Executing Enhanced Strategic Implementation Cycle")
            global strategic_implementation_cycle
            if strategic_implementation_cycle:
                collaboration_session = await strategic_implementation_cycle.execute_full_cycle(
                    requirement=requirement,
                    project_context=project_context,
                    quality_requirements=quality_requirements
                )
                session["collaboration_session"] = collaboration_session.__dict__
                session["phases"] = [{"phase": "strategic_implementation_cycle", "results": collaboration_session.__dict__}]
            else:
                # Fallback to original phases if collaboration patterns not available
                print("🧠 Phase 1: MASTERMIND Strategic Analysis")
                strategic_phase = await self._execute_strategic_analysis_phase(
                    requirement, project_context, quality_requirements, session_id
                )
                session["phases"].append(strategic_phase)
                
                print("🤝 Phase 2: Strategic Handoff to EXECUTOR")
                handoff_phase = await self._execute_strategic_handoff_phase(
                    strategic_phase["results"], session_id
                )
                session["phases"].append(handoff_phase)
                
                print("⚡ Phase 3: EXECUTOR TDD Implementation")
                implementation_phase = await self._execute_implementation_phase(
                    handoff_phase["handoff_package"], quality_requirements, session_id
                )
                session["phases"].append(implementation_phase)
                
                print("🔍 Phase 4: Quality Validation & Feedback")
                validation_phase = await self._execute_validation_phase(
                    implementation_phase["results"], handoff_phase["handoff_package"], session_id
                )
                session["phases"].append(validation_phase)
                
                print("🚀 Phase 5: Deployment & Monitoring")
                deployment_phase = await self._execute_deployment_phase(
                    validation_phase["validated_implementation"], project_context, session_id
                )
                session["phases"].append(deployment_phase)
            
            # Finalize session
            session["status"] = "completed"
            session["end_time"] = time.time()
            session["total_duration"] = session["end_time"] - cycle_start
            
            # Generate comprehensive results
            session["final_results"] = await self._generate_final_results(session)
            
            # Record session in history
            self.collaboration_history.append(session)
            
            return session["final_results"]
            
        except Exception as e:
            session["status"] = "failed"
            session["error"] = str(e)
            session["end_time"] = time.time()
            raise
        
        finally:
            # Clean up active session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
    
    async def continuous_improvement_cycle(self,
                                         codebase_path: str,
                                         improvement_targets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute continuous improvement cycle to enhance existing codebase.
        
        Args:
            codebase_path: Path to codebase for analysis
            improvement_targets: Specific improvement targets
            
        Returns:
            Dict[str, Any]: Improvement cycle results
        """
        improvement_start = time.time()
        
        # Phase 1: Comprehensive codebase analysis
        print("📊 Analyzing current codebase...")
        codebase_analysis = await codebase_analyzer.comprehensive_analysis(
            focus_areas=improvement_targets.get("focus_areas", ["all"]),
            include_dependencies=True,
            include_security=True
        )
        
        # Phase 2: Identify improvement opportunities
        print("🎯 Identifying improvement opportunities...")
        improvement_opportunities = await self._identify_improvement_opportunities(
            codebase_analysis, improvement_targets
        )
        
        # Phase 3: Prioritize improvements
        print("📈 Prioritizing improvements...")
        prioritized_improvements = await self._prioritize_improvements(
            improvement_opportunities, improvement_targets
        )
        
        # Phase 4: Execute improvements
        improvements_results = []
        for improvement in prioritized_improvements[:5]:  # Top 5 improvements
            print(f"🔧 Implementing: {improvement['title']}")
            result = await self._execute_improvement(improvement, codebase_analysis)
            improvements_results.append(result)
        
        # Phase 5: Validate improvements
        print("✅ Validating improvements...")
        validation_results = await self._validate_improvements(improvements_results)
        
        return {
            "improvement_cycle_duration": time.time() - improvement_start,
            "codebase_analysis": codebase_analysis,
            "improvement_opportunities": improvement_opportunities,
            "implemented_improvements": improvements_results,
            "validation_results": validation_results,
            "quality_impact": await self._assess_quality_impact(
                codebase_analysis, improvements_results
            )
        }
    
    async def collaborative_problem_solving(self,
                                          problem_statement: str,
                                          complexity_level: str,
                                          time_constraint: Optional[int] = None) -> Dict[str, Any]:
        """
        Facilitate collaborative problem-solving between MASTERMIND and EXECUTOR.
        
        Args:
            problem_statement: Problem to solve
            complexity_level: Complexity level (low, medium, high)
            time_constraint: Time constraint in minutes
            
        Returns:
            Dict[str, Any]: Collaborative solution results
        """
        collaboration_start = time.time()
        
        # Start collaboration session
        session_id = await communication_bus.start_collaboration_session(
            agents=[AgentRole.MASTERMIND, AgentRole.EXECUTOR],
            problem_definition=problem_statement
        )
        
        # MASTERMIND analysis contribution
        mastermind_analysis = await self.mastermind.process_task(
            self.mastermind.create_task_context(
                description=f"Analyze problem: {problem_statement}",
                requirements={"complexity": complexity_level},
                constraints={"time_limit": time_constraint} if time_constraint else {}
            )
        )
        
        await communication_hub.add_collaboration_contribution(
            session_id, AgentRole.MASTERMIND, {
                "type": "strategic_analysis",
                "content": mastermind_analysis,
                "confidence": 0.9
            }
        )
        
        # EXECUTOR implementation contribution
        executor_solution = await self.executor.process_task(
            self.executor.create_task_context(
                description=f"Implement solution for: {problem_statement}",
                requirements={"strategic_guidance": mastermind_analysis},
                constraints={"time_limit": time_constraint} if time_constraint else {}
            )
        )
        
        await communication_hub.add_collaboration_contribution(
            session_id, AgentRole.EXECUTOR, {
                "type": "implementation_solution",
                "content": executor_solution,
                "confidence": 0.85
            }
        )
        
        # Synthesize collaborative solution
        collaborative_solution = await self._synthesize_collaborative_solution(
            session_id, mastermind_analysis, executor_solution
        )
        
        return {
            "collaboration_duration": time.time() - collaboration_start,
            "session_id": session_id,
            "problem_statement": problem_statement,
            "mastermind_analysis": mastermind_analysis,
            "executor_solution": executor_solution,
            "collaborative_solution": collaborative_solution,
            "solution_quality": await self._assess_solution_quality(collaborative_solution)
        }
    
    # Core phase execution methods
    
    async def _execute_strategic_analysis_phase(self,
                                              requirement: str,
                                              project_context: Dict[str, Any],
                                              quality_requirements: Dict[str, Any],
                                              session_id: str) -> Dict[str, Any]:
        """Execute MASTERMIND strategic analysis phase."""
        phase_start = time.time()
        
        # Create task context for MASTERMIND
        task_context = self.mastermind.create_task_context(
            description=requirement,
            requirements=project_context,
            quality_gates=quality_requirements,
            architectural_context=project_context.get("architecture", {}),
            performance_targets=quality_requirements.get("performance", {}),
            security_requirements=quality_requirements.get("security", {})
        )
        
        # Execute strategic analysis
        strategic_results = await self.mastermind.process_task(task_context)
        
        return {
            "phase": "strategic_analysis",
            "duration": time.time() - phase_start,
            "task_context": task_context.__dict__,
            "results": strategic_results,
            "quality_assessment": await self._assess_strategic_quality(strategic_results)
        }
    
    async def _execute_strategic_handoff_phase(self,
                                             strategic_results: Dict[str, Any],
                                             session_id: str) -> Dict[str, Any]:
        """Execute strategic handoff to EXECUTOR."""
        phase_start = time.time()
        
        # Create handoff package
        handoff_package = await communication_hub.strategic_handoff(
            mastermind_analysis=strategic_results,
            task_context=self.mastermind.active_tasks[list(self.mastermind.active_tasks.keys())[-1]]
        )
        
        return {
            "phase": "strategic_handoff",
            "duration": time.time() - phase_start,
            "handoff_package": handoff_package,
            "handoff_quality": await self._assess_handoff_quality(handoff_package)
        }
    
    async def _execute_implementation_phase(self,
                                          handoff_package: HandoffPackage,
                                          quality_requirements: Dict[str, Any],
                                          session_id: str) -> Dict[str, Any]:
        """Execute EXECUTOR TDD implementation phase."""
        phase_start = time.time()
        
        # Create task context for EXECUTOR
        task_context = self.executor.create_task_context(
            description=handoff_package.task_context.description,
            requirements=handoff_package.implementation_guidance,
            quality_gates=handoff_package.quality_requirements,
            architectural_context=handoff_package.strategic_analysis.get("architecture_design", {}),
            performance_targets=handoff_package.quality_requirements.get("performance_requirements", {}),
            security_requirements=handoff_package.quality_requirements.get("security_requirements", {})
        )
        
        # Execute TDD implementation cycles
        implementation_results = await self._execute_tdd_cycles(
            handoff_package, quality_requirements, task_context
        )
        
        return {
            "phase": "tdd_implementation",
            "duration": time.time() - phase_start,
            "task_context": task_context.__dict__,
            "results": implementation_results,
            "tdd_compliance": await self._assess_tdd_compliance(implementation_results)
        }
    
    async def _execute_tdd_cycles(self,
                                handoff_package: HandoffPackage,
                                quality_requirements: Dict[str, Any],
                                task_context: TaskContext) -> Dict[str, Any]:
        """Execute multiple TDD cycles for complete implementation."""
        tdd_results = {
            "cycles": [],
            "final_implementation": "",
            "comprehensive_tests": "",
            "quality_metrics": {},
            "refactoring_history": []
        }
        
        # Break down requirement into implementable chunks
        implementation_chunks = await self._break_down_implementation(
            handoff_package.task_context.description,
            handoff_package.implementation_guidance
        )
        
        # Execute TDD cycle for each chunk
        for i, chunk in enumerate(implementation_chunks):
            print(f"  🔄 TDD Cycle {i+1}: {chunk['title']}")
            
            cycle_result = await code_generator.tdd_implementation_cycle(
                requirement=chunk["requirement"],
                architectural_guidance=handoff_package.implementation_guidance,
                quality_requirements=handoff_package.quality_requirements,
                iteration=i+1
            )
            
            tdd_results["cycles"].append(cycle_result)
        
        # Integrate all cycles into final implementation
        tdd_results["final_implementation"] = await self._integrate_tdd_cycles(
            tdd_results["cycles"]
        )
        
        # Generate comprehensive test suite
        tdd_results["comprehensive_tests"] = await testing_framework.comprehensive_test_execution(
            code_under_test=tdd_results["final_implementation"],
            test_requirements=handoff_package.quality_requirements,
            quality_gates=self.quality_thresholds
        )
        
        # Calculate final quality metrics
        tdd_results["quality_metrics"] = await self._calculate_implementation_quality_metrics(
            tdd_results["final_implementation"],
            tdd_results["comprehensive_tests"]
        )
        
        return tdd_results
    
    async def _execute_validation_phase(self,
                                      implementation_results: Dict[str, Any],
                                      handoff_package: HandoffPackage,
                                      session_id: str) -> Dict[str, Any]:
        """Execute quality validation and feedback phase."""
        phase_start = time.time()
        
        # MASTERMIND review of implementation
        review_results = await self.mastermind.review_executor_implementation(
            implementation_report=implementation_results
        )
        
        # Generate feedback
        feedback = await communication_hub.implementation_feedback(
            executor_results=implementation_results,
            original_handoff=handoff_package
        )
        
        # Check quality gates
        quality_gates_status = await self._check_quality_gates(
            implementation_results, self.quality_thresholds
        )
        
        return {
            "phase": "quality_validation",
            "duration": time.time() - phase_start,
            "mastermind_review": review_results,
            "feedback": feedback,
            "quality_gates_status": quality_gates_status,
            "validated_implementation": implementation_results if quality_gates_status["all_passed"] else None
        }
    
    async def _execute_deployment_phase(self,
                                      validated_implementation: Dict[str, Any],
                                      project_context: Dict[str, Any],
                                      session_id: str) -> Dict[str, Any]:
        """Execute deployment and monitoring setup phase."""
        phase_start = time.time()
        
        if not validated_implementation:
            return {
                "phase": "deployment",
                "duration": time.time() - phase_start,
                "status": "skipped",
                "reason": "Implementation failed quality validation"
            }
        
        # Generate CI/CD pipeline
        pipeline_config = await devops_automation.generate_cicd_pipeline(
            project_type=project_context.get("project_type", "web_app"),
            quality_requirements=self.quality_thresholds,
            deployment_targets=project_context.get("deployment_targets", ["development"]),
            security_requirements=project_context.get("security_requirements", {})
        )
        
        # Setup monitoring
        monitoring_config = await monitoring_metrics.comprehensive_monitoring_setup(
            application_spec=project_context.get("application_spec", {}),
            monitoring_requirements=project_context.get("monitoring_requirements", {})
        )
        
        return {
            "phase": "deployment_setup",
            "duration": time.time() - phase_start,
            "pipeline_config": pipeline_config,
            "monitoring_config": monitoring_config,
            "deployment_ready": True
        }
    
    # Helper methods
    
    def _register_mcp_tools(self):
        """Register MCP tools with both agents."""
        # MASTERMIND tools
        self.mastermind.register_tool("codebase_analyzer", codebase_analyzer.comprehensive_analysis)
        self.mastermind.register_tool("agent_communicator", communication_hub.strategic_handoff)
        
        # EXECUTOR tools  
        self.executor.register_tool("code_generator", code_generator.tdd_implementation_cycle)
        self.executor.register_tool("testing_framework", testing_framework.comprehensive_test_execution)
        self.executor.register_tool("devops_automation", devops_automation.generate_cicd_pipeline)
        self.executor.register_tool("monitoring_metrics", monitoring_metrics.comprehensive_monitoring_setup)
        
        # Shared tools
        for agent in [self.mastermind, self.executor]:
            agent.register_tool("agent_communicator", communication_hub.collaborative_session)
    
    async def _break_down_implementation(self,
                                       requirement: str,
                                       guidance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Break down implementation into TDD-manageable chunks."""
        # Simplified breakdown - would use more sophisticated analysis
        return [
            {
                "title": "Core Implementation",
                "requirement": requirement,
                "priority": "high"
            }
        ]
    
    async def _integrate_tdd_cycles(self, cycles: List[Dict[str, Any]]) -> str:
        """Integrate multiple TDD cycles into final implementation."""
        if not cycles:
            return ""
        
        # For now, return the refactored code from the last cycle
        final_cycle = cycles[-1]
        return final_cycle.refactored_code or final_cycle.implementation_code
    
    async def _calculate_implementation_quality_metrics(self,
                                                       implementation: str,
                                                       test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for implementation."""
        return {
            "test_coverage": test_results.get("coverage_analysis", {}).get("line_coverage", 0),
            "mutation_score": test_results.get("mutation_testing", {}).get("mutation_score", 0),
            "security_score": test_results.get("security_testing", {}).get("security_score", 0),
            "performance_score": test_results.get("performance_testing", {}).get("performance_score", 0),
            "maintainability_score": 8.0  # Would calculate from code analysis
        }
    
    async def _check_quality_gates(self,
                                 implementation_results: Dict[str, Any],
                                 thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Check if implementation meets quality gate thresholds."""
        quality_metrics = implementation_results.get("quality_metrics", {})
        
        gates_status = {}
        for gate, threshold in thresholds.items():
            actual_value = quality_metrics.get(gate, 0)
            gates_status[gate] = {
                "passed": actual_value >= threshold,
                "actual": actual_value,
                "threshold": threshold
            }
        
        all_passed = all(status["passed"] for status in gates_status.values())
        
        return {
            "all_passed": all_passed,
            "individual_gates": gates_status,
            "passed_count": sum(1 for status in gates_status.values() if status["passed"]),
            "total_gates": len(gates_status)
        }
    
    async def _generate_final_results(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final results for development cycle."""
        return {
            "session_id": session["session_id"],
            "requirement": session["requirement"],
            "total_duration": session["total_duration"],
            "phases_executed": len(session["phases"]),
            "quality_assessment": await self._assess_overall_quality(session),
            "deliverables": await self._extract_deliverables(session),
            "metrics": await self._extract_session_metrics(session),
            "recommendations": await self._generate_session_recommendations(session)
        }
    
    # Assessment methods (placeholders for complex implementations)
    
    async def _assess_strategic_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {"strategic_quality_score": 8.5}
    
    async def _assess_handoff_quality(self, handoff: HandoffPackage) -> Dict[str, Any]:
        return {"handoff_quality_score": 9.0}
    
    async def _assess_tdd_compliance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {"tdd_compliance_score": 9.5}
    
    async def _assess_overall_quality(self, session: Dict[str, Any]) -> Dict[str, Any]:
        return {"overall_quality_score": 9.0}
    
    async def _extract_deliverables(self, session: Dict[str, Any]) -> Dict[str, Any]:
        return {"deliverables": ["implementation", "tests", "documentation"]}
    
    async def _extract_session_metrics(self, session: Dict[str, Any]) -> Dict[str, Any]:
        return {"session_metrics": {}}
    
    async def _generate_session_recommendations(self, session: Dict[str, Any]) -> List[str]:
        return ["Continue with deployment", "Add more integration tests"]


# Global orchestrator instance
orchestrator = AgentOrchestrator()