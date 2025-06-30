#!/usr/bin/env python3
"""
SPARC Trio Quality Standards - London School TDD Approach
Defines 9.5+/10 quality standards for cross-validation
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum


class QualityLevel(Enum):
    """Quality levels for SPARC outputs"""
    UNACCEPTABLE = "unacceptable"  # 0-5
    POOR = "poor"                  # 5-6
    ACCEPTABLE = "acceptable"      # 6-7
    GOOD = "good"                  # 7-8
    VERY_GOOD = "very_good"        # 8-9
    EXCELLENT = "excellent"        # 9-9.5
    EXCEPTIONAL = "exceptional"    # 9.5-10


@dataclass
class QualityMetric:
    """Individual quality metric definition"""
    name: str
    description: str
    weight: float
    min_score: float
    target_score: float
    validation_criteria: List[str]


@dataclass
class AgentQualityStandard:
    """Quality standards for a SPARC agent"""
    agent_name: str
    overall_target: float
    metrics: List[QualityMetric]
    mandatory_requirements: List[str]
    quality_gates: List[str]


class SPARCQualityStandards:
    """Comprehensive quality standards for SPARC trio"""
    
    def __init__(self):
        self.researcher_standard = self._define_researcher_standard()
        self.mastermind_standard = self._define_mastermind_standard()
        self.executor_standard = self._define_executor_standard()
    
    def _define_researcher_standard(self) -> AgentQualityStandard:
        """Define 9.5+/10 quality standard for RESEARCHER agent"""
        
        metrics = [
            QualityMetric(
                name="source_diversity",
                description="Use of multiple, credible sources for research",
                weight=0.20,
                min_score=9.0,
                target_score=9.5,
                validation_criteria=[
                    "Minimum 5 distinct sources cited",
                    "Mix of academic, industry, and primary sources",
                    "Sources are current (within 2 years for tech topics)",
                    "Source credibility is established"
                ]
            ),
            QualityMetric(
                name="analysis_depth",
                description="Comprehensive and thorough analysis",
                weight=0.25,
                min_score=9.0,
                target_score=9.5,
                validation_criteria=[
                    "Multi-dimensional analysis (technical, business, risk)",
                    "Identification of potential blind spots",
                    "Evidence-based conclusions with confidence scoring",
                    "Consideration of alternative approaches"
                ]
            ),
            QualityMetric(
                name="actionable_insights",
                description="Clear, implementable recommendations",
                weight=0.20,
                min_score=9.0,
                target_score=9.5,
                validation_criteria=[
                    "Specific, actionable recommendations",
                    "Implementation guidance provided",
                    "Risk mitigation strategies included",
                    "Success metrics defined"
                ]
            ),
            QualityMetric(
                name="market_intelligence",
                description="Current market trends and competitive analysis",
                weight=0.15,
                min_score=8.5,
                target_score=9.0,
                validation_criteria=[
                    "Current market trends identified",
                    "Competitive landscape analysis",
                    "Technology adoption patterns",
                    "Industry best practices referenced"
                ]
            ),
            QualityMetric(
                name="evidence_quality",
                description="Quality and reliability of supporting evidence",
                weight=0.20,
                min_score=9.0,
                target_score=9.5,
                validation_criteria=[
                    "Quantitative data where possible",
                    "Peer-reviewed sources prioritized",
                    "Evidence directly supports conclusions",
                    "Confidence levels explicitly stated"
                ]
            )
        ]
        
        mandatory_requirements = [
            "Minimum 5 credible sources cited with URLs/references",
            "Confidence score (85-95%) for each major conclusion",
            "Risk analysis with mitigation strategies",
            "Actionable next steps with timelines",
            "Alternative approaches considered and evaluated"
        ]
        
        quality_gates = [
            "Overall score ≥ 9.5/10 from cross-validation",
            "All mandatory requirements met",
            "No factual errors or unsupported claims",
            "Evidence directly supports all conclusions",
            "Recommendations are specific and implementable"
        ]
        
        return AgentQualityStandard(
            agent_name="RESEARCHER",
            overall_target=9.5,
            metrics=metrics,
            mandatory_requirements=mandatory_requirements,
            quality_gates=quality_gates
        )
    
    def _define_mastermind_standard(self) -> AgentQualityStandard:
        """Define 9.5+/10 quality standard for MASTERMIND agent"""
        
        metrics = [
            QualityMetric(
                name="architectural_completeness",
                description="Complete and scalable system architecture",
                weight=0.25,
                min_score=9.0,
                target_score=9.5,
                validation_criteria=[
                    "All system components defined",
                    "Data flow and integration patterns specified",
                    "Scalability considerations addressed",
                    "Technology stack justified with alternatives"
                ]
            ),
            QualityMetric(
                name="risk_assessment",
                description="Comprehensive risk analysis and mitigation",
                weight=0.20,
                min_score=9.0,
                target_score=9.5,
                validation_criteria=[
                    "Technical risks identified and prioritized",
                    "Business risks evaluated",
                    "Mitigation strategies for each risk",
                    "Contingency plans defined"
                ]
            ),
            QualityMetric(
                name="implementation_strategy",
                description="Clear implementation roadmap and strategy",
                weight=0.20,
                min_score=9.0,
                target_score=9.5,
                validation_criteria=[
                    "Phased implementation approach",
                    "Dependencies and prerequisites identified",
                    "Resource requirements estimated",
                    "Timeline with milestones defined"
                ]
            ),
            QualityMetric(
                name="performance_design",
                description="Performance requirements and optimization strategy",
                weight=0.15,
                min_score=8.5,
                target_score=9.0,
                validation_criteria=[
                    "Performance targets specified (latency, throughput)",
                    "Optimization strategies defined",
                    "Monitoring and alerting approach",
                    "Scalability patterns implemented"
                ]
            ),
            QualityMetric(
                name="security_architecture",
                description="Security-first design with compliance considerations",
                weight=0.20,
                min_score=9.0,
                target_score=9.5,
                validation_criteria=[
                    "Security architecture defined",
                    "Threat model considerations",
                    "Compliance requirements addressed",
                    "Security monitoring and incident response"
                ]
            )
        ]
        
        mandatory_requirements = [
            "Complete system architecture diagram/description",
            "Technology stack with justification and alternatives",
            "Performance targets (latency, throughput, availability)",
            "Security architecture with threat model",
            "Risk assessment with mitigation strategies",
            "Implementation roadmap with phases and timelines"
        ]
        
        quality_gates = [
            "Overall score ≥ 9.5/10 from cross-validation",
            "Architecture supports stated requirements",
            "All non-functional requirements addressed",
            "Implementation strategy is realistic and detailed",
            "Security and compliance requirements met"
        ]
        
        return AgentQualityStandard(
            agent_name="MASTERMIND",
            overall_target=9.5,
            metrics=metrics,
            mandatory_requirements=mandatory_requirements,
            quality_gates=quality_gates
        )
    
    def _define_executor_standard(self) -> AgentQualityStandard:
        """Define 9.5+/10 quality standard for EXECUTOR agent"""
        
        metrics = [
            QualityMetric(
                name="implementation_quality",
                description="Production-ready code with best practices",
                weight=0.25,
                min_score=9.0,
                target_score=9.5,
                validation_criteria=[
                    "TDD approach with comprehensive test coverage (95%+)",
                    "Security-first development (OWASP compliance)",
                    "Code quality metrics met (complexity, maintainability)",
                    "Error handling and logging implemented"
                ]
            ),
            QualityMetric(
                name="testing_strategy",
                description="Comprehensive testing across all levels",
                weight=0.20,
                min_score=9.0,
                target_score=9.5,
                validation_criteria=[
                    "Unit tests (70% of test pyramid)",
                    "Integration tests (20% of test pyramid)",
                    "End-to-end tests (10% of test pyramid)",
                    "Performance and security testing included"
                ]
            ),
            QualityMetric(
                name="devops_automation",
                description="Complete CI/CD pipeline and automation",
                weight=0.20,
                min_score=9.0,
                target_score=9.5,
                validation_criteria=[
                    "Automated build and deployment pipeline",
                    "Quality gates in CI/CD (tests, security, performance)",
                    "Monitoring and observability setup",
                    "Rollback and disaster recovery procedures"
                ]
            ),
            QualityMetric(
                name="production_readiness",
                description="Production deployment and operational excellence",
                weight=0.15,
                min_score=8.5,
                target_score=9.0,
                validation_criteria=[
                    "Containerization and orchestration ready",
                    "Environment configuration management",
                    "Health checks and readiness probes",
                    "Performance optimization implemented"
                ]
            ),
            QualityMetric(
                name="operational_excellence",
                description="Monitoring, alerting, and maintenance procedures",
                weight=0.20,
                min_score=9.0,
                target_score=9.5,
                validation_criteria=[
                    "Comprehensive monitoring and alerting",
                    "Log aggregation and analysis setup",
                    "Performance metrics and dashboards",
                    "Incident response procedures documented"
                ]
            )
        ]
        
        mandatory_requirements = [
            "Test coverage ≥ 95% with mutation testing score ≥ 90%",
            "Security scan with zero critical vulnerabilities",
            "Performance benchmarks meet requirements (<50ms response)",
            "Complete CI/CD pipeline with quality gates",
            "Production monitoring and alerting configured",
            "Comprehensive documentation and runbooks"
        ]
        
        quality_gates = [
            "Overall score ≥ 9.5/10 from cross-validation",
            "All tests pass (100% pass rate)",
            "Security compliance verified",
            "Performance benchmarks met",
            "Production deployment ready"
        ]
        
        return AgentQualityStandard(
            agent_name="EXECUTOR",
            overall_target=9.5,
            metrics=metrics,
            mandatory_requirements=mandatory_requirements,
            quality_gates=quality_gates
        )
    
    def get_quality_standard(self, agent_name: str) -> Optional[AgentQualityStandard]:
        """Get quality standard for specific agent"""
        standards = {
            "RESEARCHER": self.researcher_standard,
            "MASTERMIND": self.mastermind_standard,
            "EXECUTOR": self.executor_standard
        }
        return standards.get(agent_name.upper())
    
    def get_all_standards(self) -> Dict[str, AgentQualityStandard]:
        """Get all quality standards"""
        return {
            "RESEARCHER": self.researcher_standard,
            "MASTERMIND": self.mastermind_standard,
            "EXECUTOR": self.executor_standard
        }
    
    def calculate_weighted_score(self, agent_name: str, metric_scores: Dict[str, float]) -> float:
        """Calculate weighted quality score for an agent"""
        standard = self.get_quality_standard(agent_name)
        if not standard:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric in standard.metrics:
            if metric.name in metric_scores:
                score = metric_scores[metric.name]
                total_score += score * metric.weight
                total_weight += metric.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def validate_quality_gates(self, agent_name: str, output_analysis: Dict[str, Any]) -> Dict[str, bool]:
        """Validate quality gates for an agent"""
        standard = self.get_quality_standard(agent_name)
        if not standard:
            return {}
        
        gate_results = {}
        
        for gate in standard.quality_gates:
            # This would be implemented based on specific validation logic
            # For now, returning placeholder implementation
            gate_results[gate] = True  # Placeholder
        
        return gate_results


def get_quality_improvement_prompt(agent_name: str, current_score: float, 
                                 failed_criteria: List[str]) -> str:
    """Generate improvement prompt based on quality standards"""
    standards = SPARCQualityStandards()
    standard = standards.get_quality_standard(agent_name)
    
    if not standard:
        return f"No quality standard found for {agent_name}"
    
    prompt = f"""
# Quality Improvement Required for {agent_name} Agent

## Current Performance
- Score: {current_score:.1f}/10 (Target: {standard.overall_target}/10)
- Status: {'UNACCEPTABLE' if current_score < 9.0 else 'NEEDS IMPROVEMENT'}

## Failed Quality Criteria
{chr(10).join(f'- {criteria}' for criteria in failed_criteria)}

## Required Quality Standards
{chr(10).join(f'- {req}' for req in standard.mandatory_requirements)}

## Quality Metrics to Improve
{chr(10).join(f'- **{metric.name.title().replace("_", " ")}** (Weight: {metric.weight:.0%}, Target: {metric.target_score}/10)' for metric in standard.metrics)}

## Specific Improvement Actions
To achieve {standard.overall_target}/10:

1. **Address Failed Criteria**: Focus on the specific areas identified above
2. **Meet All Mandatory Requirements**: Ensure every requirement is fully satisfied
3. **Exceed Quality Metrics**: Aim for target scores in each weighted metric
4. **Validate with Cross-Validation**: Test output against OpenAI validation before submission

## Quality Gates
All of the following must be met:
{chr(10).join(f'- {gate}' for gate in standard.quality_gates)}

Please revise your {agent_name.lower()} output to meet these standards.
"""
    
    return prompt


if __name__ == "__main__":
    standards = SPARCQualityStandards()
    
    print("🏆 SPARC Trio Quality Standards (Target: 9.5+/10)")
    print("=" * 60)
    
    for agent_name, standard in standards.get_all_standards().items():
        print(f"\n{agent_name} Agent:")
        print(f"  Target Score: {standard.overall_target}/10")
        print(f"  Quality Metrics: {len(standard.metrics)}")
        print(f"  Mandatory Requirements: {len(standard.mandatory_requirements)}")
        print(f"  Quality Gates: {len(standard.quality_gates)}")
    
    print(f"\n✅ Quality standards defined for London School TDD approach")