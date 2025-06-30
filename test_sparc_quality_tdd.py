#!/usr/bin/env python3
"""
London School TDD for SPARC Quality - BDD Test Framework
Outside-in testing for 9.5+/10 quality requirements
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List
from dataclasses import dataclass
from fire_cross_validation import FireCrossValidator
from sparc_quality_standards import SPARCQualityStandards, QualityLevel


@dataclass
class QualityTestScenario:
    """BDD scenario for quality testing"""
    name: str
    agent: str
    given: List[str]
    when: str
    then: List[str]
    expected_score: float
    quality_requirements: List[str]


class SPARCQualityTDD:
    """London School TDD framework for SPARC quality improvement"""
    
    def __init__(self):
        self.cross_validator = FireCrossValidator()
        self.quality_standards = SPARCQualityStandards()
        self.current_scenarios: List[QualityTestScenario] = []
        self.agent_mocks = {
            "RESEARCHER": Mock(),
            "MASTERMIND": Mock(), 
            "EXECUTOR": Mock()
        }
    
    def scenario(self, name: str, agent: str) -> 'QualityTestBuilder':
        """Start defining a quality test scenario"""
        return QualityTestBuilder(self, name, agent)
    
    def add_scenario(self, scenario: QualityTestScenario):
        """Add scenario to test suite"""
        self.current_scenarios.append(scenario)
    
    async def test_quality_scenario(self, scenario: QualityTestScenario) -> Dict[str, Any]:
        """Execute a quality test scenario"""
        result = {
            'scenario_name': scenario.name,
            'agent': scenario.agent,
            'expected_score': scenario.expected_score,
            'actual_score': 0.0,
            'status': 'FAILED',
            'validation_feedback': '',
            'failed_requirements': [],
            'errors': []
        }
        
        try:
            # Mock agent output (initially poor quality for TDD red phase)
            mock_output = self._generate_mock_output(scenario.agent, scenario.quality_requirements)
            
            # Cross-validate the output
            validation_result = self.cross_validator.cross_validate(
                mock_output, 
                scenario.agent.lower()
            )
            
            if validation_result['status'] == 'success':
                result['actual_score'] = validation_result['validation_score']
                result['validation_feedback'] = validation_result['feedback']
                
                # Check if meets quality expectations
                if result['actual_score'] >= scenario.expected_score:
                    result['status'] = 'PASSED'
                else:
                    result['status'] = 'FAILED'
                    result['failed_requirements'] = self._identify_failed_requirements(
                        scenario, validation_result['full_feedback']
                    )
            else:
                result['errors'].append(f"Cross-validation failed: {validation_result.get('message', 'Unknown error')}")
        
        except Exception as e:
            result['errors'].append(str(e))
        
        return result
    
    def _generate_mock_output(self, agent: str, requirements: List[str]) -> str:
        """Generate mock output for testing (initially poor quality)"""
        
        # Poor quality outputs to drive TDD red phase
        poor_outputs = {
            "RESEARCHER": """
            Basic Research Analysis: Trading Algorithm
            
            I looked at some trading strategies and think momentum indicators might work.
            RSI and MACD are popular. Some backtesting shows decent results.
            """,
            
            "MASTERMIND": """
            System Design: Trading Platform
            
            We should use Python and FastAPI. Need a database for data.
            Microservices architecture would be good. Deploy with Docker.
            """,
            
            "EXECUTOR": """
            Implementation Plan: Trading System
            
            Phase 1: Write some code
            Phase 2: Add tests
            Phase 3: Deploy
            
            Should be pretty straightforward.
            """
        }
        
        return poor_outputs.get(agent, "Generic poor quality output")
    
    def _identify_failed_requirements(self, scenario: QualityTestScenario, 
                                   feedback: str) -> List[str]:
        """Identify which requirements failed based on validation feedback"""
        failed = []
        
        # Parse feedback to identify missing elements
        feedback_lower = feedback.lower()
        
        requirement_keywords = {
            "multiple sources": ["source", "citation", "reference"],
            "evidence-based": ["evidence", "data", "proof"],
            "risk analysis": ["risk", "mitigation", "threat"],
            "implementation details": ["detail", "specific", "implementation"],
            "performance targets": ["performance", "latency", "throughput"],
            "security considerations": ["security", "compliance", "authentication"],
            "testing strategy": ["test", "coverage", "quality"]
        }
        
        for requirement, keywords in requirement_keywords.items():
            if requirement in scenario.quality_requirements:
                if not any(keyword in feedback_lower for keyword in keywords):
                    failed.append(requirement)
        
        return failed
    
    async def run_all_quality_tests(self) -> Dict[str, Any]:
        """Run all quality test scenarios"""
        results = {
            'total_scenarios': len(self.current_scenarios),
            'passed': 0,
            'failed': 0,
            'scenario_results': [],
            'overall_status': 'FAILED'
        }
        
        for scenario in self.current_scenarios:
            scenario_result = await self.test_quality_scenario(scenario)
            results['scenario_results'].append(scenario_result)
            
            if scenario_result['status'] == 'PASSED':
                results['passed'] += 1
            else:
                results['failed'] += 1
        
        # Overall status based on all scenarios passing
        if results['failed'] == 0 and results['passed'] > 0:
            results['overall_status'] = 'PASSED'
        
        return results


class QualityTestBuilder:
    """Builder for quality test scenarios"""
    
    def __init__(self, tdd_framework: SPARCQualityTDD, name: str, agent: str):
        self.tdd_framework = tdd_framework
        self.name = name
        self.agent = agent
        self.given_conditions: List[str] = []
        self.when_action = ""
        self.then_expectations: List[str] = []
        self.expected_score = 9.5
        self.quality_requirements: List[str] = []
    
    def given(self, condition: str) -> 'QualityTestBuilder':
        """Add precondition"""
        self.given_conditions.append(condition)
        return self
    
    def when(self, action: str) -> 'QualityTestBuilder':
        """Set action"""
        self.when_action = action
        return self
    
    def then(self, expectation: str) -> 'QualityTestBuilder':
        """Add expectation"""
        self.then_expectations.append(expectation)
        return self
    
    def expecting_score_of(self, score: float) -> 'QualityTestBuilder':
        """Set expected quality score"""
        self.expected_score = score
        return self
    
    def requiring(self, requirement: str) -> 'QualityTestBuilder':
        """Add quality requirement"""
        self.quality_requirements.append(requirement)
        return self
    
    def build(self) -> QualityTestScenario:
        """Build and register the scenario"""
        scenario = QualityTestScenario(
            name=self.name,
            agent=self.agent,
            given=self.given_conditions,
            when=self.when_action,
            then=self.then_expectations,
            expected_score=self.expected_score,
            quality_requirements=self.quality_requirements
        )
        
        self.tdd_framework.add_scenario(scenario)
        return scenario


# TDD Test Cases - These should FAIL initially (Red Phase)

@pytest.fixture
def sparc_quality_tdd():
    """Fixture providing SPARC quality TDD framework"""
    return SPARCQualityTDD()


@pytest.mark.asyncio
async def test_researcher_exceptional_quality(sparc_quality_tdd):
    """Red Phase: Test RESEARCHER for 9.5+/10 quality (should FAIL initially)"""
    
    scenario = (sparc_quality_tdd
                .scenario("researcher_exceptional_analysis", "RESEARCHER")
                .given("A complex trading algorithm analysis request")
                .given("Access to market data and research sources")
                .when("RESEARCHER analyzes trading algorithm requirements")
                .then("Output should include 5+ credible sources")
                .then("Analysis should include risk assessment")
                .then("Recommendations should be actionable and specific")
                .then("Confidence scores should be provided for conclusions")
                .expecting_score_of(9.5)
                .requiring("multiple sources")
                .requiring("evidence-based")
                .requiring("risk analysis")
                .requiring("actionable recommendations")
                .build())
    
    result = await sparc_quality_tdd.test_quality_scenario(scenario)
    
    # This should FAIL in Red phase - driving improvement
    assert result['status'] == 'FAILED', "RESEARCHER should fail quality test initially (TDD Red phase)"
    assert result['actual_score'] < 9.5, f"Score {result['actual_score']} should be below 9.5 target"
    assert len(result['failed_requirements']) > 0, "Should have failed requirements to drive improvement"


@pytest.mark.asyncio
async def test_mastermind_exceptional_quality(sparc_quality_tdd):
    """Red Phase: Test MASTERMIND for 9.5+/10 quality (should FAIL initially)"""
    
    scenario = (sparc_quality_tdd
                .scenario("mastermind_complete_architecture", "MASTERMIND")
                .given("Research findings from RESEARCHER")
                .given("System requirements and constraints")
                .when("MASTERMIND designs system architecture")
                .then("Architecture should be complete and scalable")
                .then("Technology stack should be justified")
                .then("Performance targets should be specified")
                .then("Security architecture should be included")
                .then("Risk assessment with mitigation strategies")
                .expecting_score_of(9.5)
                .requiring("complete architecture")
                .requiring("performance targets")
                .requiring("security considerations")
                .requiring("risk analysis")
                .build())
    
    result = await sparc_quality_tdd.test_quality_scenario(scenario)
    
    # This should FAIL in Red phase - driving improvement
    assert result['status'] == 'FAILED', "MASTERMIND should fail quality test initially (TDD Red phase)"
    assert result['actual_score'] < 9.5, f"Score {result['actual_score']} should be below 9.5 target"


@pytest.mark.asyncio
async def test_executor_exceptional_quality(sparc_quality_tdd):
    """Red Phase: Test EXECUTOR for 9.5+/10 quality (should FAIL initially)"""
    
    scenario = (sparc_quality_tdd
                .scenario("executor_production_ready", "EXECUTOR")
                .given("System architecture from MASTERMIND")
                .given("Implementation requirements and constraints")
                .when("EXECUTOR creates implementation plan")
                .then("Implementation should follow TDD approach")
                .then("Test coverage should be 95%+")
                .then("Security compliance should be addressed")
                .then("CI/CD pipeline should be defined")
                .then("Monitoring and observability should be included")
                .expecting_score_of(9.5)
                .requiring("testing strategy")
                .requiring("security considerations")
                .requiring("implementation details")
                .requiring("devops automation")
                .build())
    
    result = await sparc_quality_tdd.test_quality_scenario(scenario)
    
    # This should FAIL in Red phase - driving improvement
    assert result['status'] == 'FAILED', "EXECUTOR should fail quality test initially (TDD Red phase)"
    assert result['actual_score'] < 9.5, f"Score {result['actual_score']} should be below 9.5 target"


@pytest.mark.asyncio
async def test_full_sparc_trio_exceptional_quality(sparc_quality_tdd):
    """Red Phase: Test complete SPARC trio for 9.5+/10 average (should FAIL initially)"""
    
    # Test all three agents
    researcher_scenario = (sparc_quality_tdd
                          .scenario("trio_researcher", "RESEARCHER")
                          .expecting_score_of(9.5)
                          .requiring("comprehensive analysis")
                          .build())
    
    mastermind_scenario = (sparc_quality_tdd
                          .scenario("trio_mastermind", "MASTERMIND")
                          .expecting_score_of(9.5)
                          .requiring("complete architecture")
                          .build())
    
    executor_scenario = (sparc_quality_tdd
                        .scenario("trio_executor", "EXECUTOR")
                        .expecting_score_of(9.5)
                        .requiring("production ready")
                        .build())
    
    results = await sparc_quality_tdd.run_all_quality_tests()
    
    # Calculate average score
    total_score = sum(r['actual_score'] for r in results['scenario_results'])
    average_score = total_score / len(results['scenario_results']) if results['scenario_results'] else 0
    
    # This should FAIL in Red phase - driving trio improvement
    assert results['overall_status'] == 'FAILED', "SPARC trio should fail quality test initially (TDD Red phase)"
    assert average_score < 9.5, f"Average score {average_score:.1f} should be below 9.5 target"
    assert results['failed'] > 0, "Should have failed tests to drive improvement"


def main():
    """Run TDD quality tests"""
    print("🔴 London School TDD - Red Phase: Quality Tests (Should FAIL)")
    print("=" * 60)
    
    tdd = SPARCQualityTDD()
    
    # These tests should all FAIL initially, driving the implementation
    scenarios = [
        ("RESEARCHER Exceptional Quality", "researcher_exceptional_analysis", "RESEARCHER"),
        ("MASTERMIND Complete Architecture", "mastermind_complete_architecture", "MASTERMIND"),
        ("EXECUTOR Production Ready", "executor_production_ready", "EXECUTOR")
    ]
    
    async def run_tests():
        for test_name, scenario_name, agent in scenarios:
            print(f"\n🧪 Testing: {test_name}")
            
            scenario = (tdd.scenario(scenario_name, agent)
                       .expecting_score_of(9.5)
                       .requiring("exceptional quality")
                       .build())
            
            result = await tdd.test_quality_scenario(scenario)
            
            status_emoji = "❌" if result['status'] == 'FAILED' else "✅"
            print(f"{status_emoji} {test_name}: {result['actual_score']:.1f}/10 (Target: 9.5/10)")
            
            if result['failed_requirements']:
                print(f"   Failed Requirements: {', '.join(result['failed_requirements'])}")
    
    asyncio.run(run_tests())
    print(f"\n🔴 Red Phase Complete - All tests should FAIL, driving quality improvements")


if __name__ == "__main__":
    main()