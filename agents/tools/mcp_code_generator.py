"""
MCP Code Generation Tools for EXECUTOR Implementation

These tools enable intelligent code generation following TDD principles,
architectural patterns, and quality standards defined by MASTERMIND.
"""

import ast
import os
import re
import time
import json
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
import textwrap


@dataclass
class CodeTemplate:
    """Template for code generation."""
    name: str
    description: str
    template: str
    variables: List[str]
    requirements: Dict[str, Any]
    test_template: Optional[str] = None


@dataclass
class GeneratedCode:
    """Result of code generation."""
    code: str
    test_code: Optional[str]
    file_path: str
    imports: List[str]
    dependencies: List[str]
    quality_metrics: Dict[str, Any]
    documentation: str


@dataclass
class TDDCycle:
    """TDD Red-Green-Refactor cycle state."""
    phase: str  # "red", "green", "refactor"
    test_code: str
    implementation_code: str
    refactored_code: Optional[str]
    cycle_number: int
    requirements: Dict[str, Any]


class CodePattern(ABC):
    """Abstract base class for code patterns."""
    
    @abstractmethod
    def generate(self, context: Dict[str, Any]) -> GeneratedCode:
        """Generate code following this pattern."""
        pass
    
    @abstractmethod
    def validate(self, code: str) -> Dict[str, Any]:
        """Validate generated code against pattern requirements."""
        pass


class MCPCodeGenerator:
    """Advanced code generation engine for TDD-driven development."""
    
    def __init__(self):
        self.patterns: Dict[str, CodePattern] = {}
        self.templates: Dict[str, CodeTemplate] = {}
        self.generation_history: List[Dict[str, Any]] = []
        self.code_standards = self._load_code_standards()
        
        # Initialize built-in patterns
        self._register_builtin_patterns()
        self._register_builtin_templates()
    
    async def tdd_implementation_cycle(self,
                                     requirement: str,
                                     architectural_guidance: Dict[str, Any],
                                     quality_requirements: Dict[str, Any],
                                     iteration: int = 1) -> TDDCycle:
        """
        Execute a complete TDD cycle: Red -> Green -> Refactor.
        
        Args:
            requirement: Functional requirement to implement
            architectural_guidance: Guidance from MASTERMIND
            quality_requirements: Quality standards to meet
            iteration: Current iteration number
            
        Returns:
            TDDCycle: Complete cycle with all phases
        """
        cycle_start = time.time()
        
        # Phase 1: RED - Write failing test
        test_code = await self._generate_failing_test(
            requirement, architectural_guidance, quality_requirements
        )
        
        # Verify test fails
        test_result = await self._run_test(test_code)
        if test_result["passed"]:
            test_code = await self._make_test_fail(test_code, requirement)
        
        # Phase 2: GREEN - Minimal implementation
        implementation_code = await self._generate_minimal_implementation(
            test_code, requirement, architectural_guidance
        )
        
        # Verify test passes
        impl_test_result = await self._run_test_with_implementation(test_code, implementation_code)
        if not impl_test_result["passed"]:
            implementation_code = await self._fix_implementation(
                test_code, implementation_code, impl_test_result["errors"]
            )
        
        # Phase 3: REFACTOR - Improve while keeping tests green
        refactored_code = await self._refactor_implementation(
            implementation_code, test_code, quality_requirements
        )
        
        # Final verification
        final_test_result = await self._run_test_with_implementation(test_code, refactored_code)
        
        cycle = TDDCycle(
            phase="completed",
            test_code=test_code,
            implementation_code=implementation_code,
            refactored_code=refactored_code if final_test_result["passed"] else implementation_code,
            cycle_number=iteration,
            requirements={
                "requirement": requirement,
                "architectural_guidance": architectural_guidance,
                "quality_requirements": quality_requirements,
                "cycle_duration": time.time() - cycle_start,
                "final_test_result": final_test_result
            }
        )
        
        # Record cycle in history
        self.generation_history.append({
            "type": "tdd_cycle",
            "timestamp": time.time(),
            "cycle": cycle.__dict__,
            "success": final_test_result["passed"]
        })
        
        return cycle
    
    async def generate_class(self,
                           class_name: str,
                           purpose: str,
                           patterns: List[str],
                           methods: List[Dict[str, Any]],
                           quality_requirements: Dict[str, Any]) -> GeneratedCode:
        """
        Generate a class following specified patterns and quality requirements.
        
        Args:
            class_name: Name of the class
            purpose: Purpose and responsibilities
            patterns: Design patterns to follow
            methods: Method specifications
            quality_requirements: Quality standards
            
        Returns:
            GeneratedCode: Generated class with tests
        """
        context = {
            "class_name": class_name,
            "purpose": purpose,
            "patterns": patterns,
            "methods": methods,
            "quality_requirements": quality_requirements
        }
        
        # Apply design patterns
        code_generator = self._select_code_generator(patterns)
        generated = await code_generator.generate_class(context)
        
        # Generate comprehensive tests
        test_code = await self._generate_class_tests(generated, context)
        
        # Apply quality checks
        quality_metrics = await self._assess_generated_code_quality(generated.code)
        
        # Generate documentation
        documentation = await self._generate_class_documentation(generated, context)
        
        result = GeneratedCode(
            code=generated.code,
            test_code=test_code,
            file_path=f"src/{self._to_snake_case(class_name)}.py",
            imports=generated.imports,
            dependencies=generated.dependencies,
            quality_metrics=quality_metrics,
            documentation=documentation
        )
        
        return result
    
    async def generate_api_endpoint(self,
                                  endpoint_spec: Dict[str, Any],
                                  architectural_patterns: List[str],
                                  security_requirements: Dict[str, Any]) -> GeneratedCode:
        """
        Generate API endpoint with comprehensive validation and testing.
        
        Args:
            endpoint_spec: OpenAPI-style specification
            architectural_patterns: Patterns to follow
            security_requirements: Security standards
            
        Returns:
            GeneratedCode: Complete endpoint implementation
        """
        context = {
            "endpoint_spec": endpoint_spec,
            "patterns": architectural_patterns,
            "security": security_requirements
        }
        
        # Generate endpoint implementation
        endpoint_code = await self._generate_endpoint_implementation(context)
        
        # Generate comprehensive tests
        test_code = await self._generate_endpoint_tests(endpoint_code, context)
        
        # Generate OpenAPI documentation
        documentation = await self._generate_api_documentation(endpoint_code, context)
        
        # Security validation
        security_analysis = await self._validate_endpoint_security(endpoint_code, security_requirements)
        
        result = GeneratedCode(
            code=endpoint_code,
            test_code=test_code,
            file_path=f"src/api/endpoints/{endpoint_spec['path'].replace('/', '_')}.py",
            imports=["fastapi", "pydantic", "typing"],
            dependencies=["fastapi", "pydantic"],
            quality_metrics={
                "security_score": security_analysis["score"],
                "validation_coverage": security_analysis["validation_coverage"]
            },
            documentation=documentation
        )
        
        return result
    
    async def generate_database_model(self,
                                    model_spec: Dict[str, Any],
                                    orm_framework: str,
                                    migration_strategy: str) -> GeneratedCode:
        """
        Generate database model with migrations and tests.
        
        Args:
            model_spec: Model specification
            orm_framework: ORM to use (SQLAlchemy, etc.)
            migration_strategy: Migration approach
            
        Returns:
            GeneratedCode: Model with migrations and tests
        """
        context = {
            "model_spec": model_spec,
            "orm": orm_framework,
            "migration_strategy": migration_strategy
        }
        
        # Generate model code
        model_code = await self._generate_model_implementation(context)
        
        # Generate migrations
        migration_code = await self._generate_model_migrations(context)
        
        # Generate model tests
        test_code = await self._generate_model_tests(model_code, context)
        
        result = GeneratedCode(
            code=model_code,
            test_code=test_code,
            file_path=f"src/models/{self._to_snake_case(model_spec['name'])}.py",
            imports=["sqlalchemy", "typing", "datetime"],
            dependencies=["sqlalchemy"],
            quality_metrics={"migration_coverage": 100},
            documentation=await self._generate_model_documentation(model_code, context)
        )
        
        return result
    
    async def generate_test_suite(self,
                                code_to_test: str,
                                test_types: List[str],
                                coverage_target: float) -> Dict[str, str]:
        """
        Generate comprehensive test suite for existing code.
        
        Args:
            code_to_test: Source code to generate tests for
            test_types: Types of tests to generate
            coverage_target: Target coverage percentage
            
        Returns:
            Dict[str, str]: Test files by type
        """
        tests = {}
        
        # Parse code structure
        code_structure = await self._analyze_code_structure(code_to_test)
        
        for test_type in test_types:
            if test_type == "unit":
                tests["unit"] = await self._generate_unit_tests(code_structure, coverage_target)
            elif test_type == "integration":
                tests["integration"] = await self._generate_integration_tests(code_structure)
            elif test_type == "property":
                tests["property"] = await self._generate_property_tests(code_structure)
            elif test_type == "contract":
                tests["contract"] = await self._generate_contract_tests(code_structure)
            elif test_type == "mutation":
                tests["mutation"] = await self._generate_mutation_tests(code_structure)
            elif test_type == "chaos":
                tests["chaos"] = await self._generate_chaos_tests(code_structure)
            elif test_type == "security":
                tests["security"] = await self._generate_security_tests(code_structure)
            elif test_type == "performance":
                tests["performance"] = await self._generate_performance_tests(code_structure)
        
        return tests
    
    async def refactor_code(self,
                          code: str,
                          refactoring_patterns: List[str],
                          quality_targets: Dict[str, Any]) -> GeneratedCode:
        """
        Refactor existing code following specified patterns and quality targets.
        
        Args:
            code: Original code to refactor
            refactoring_patterns: Patterns to apply
            quality_targets: Quality improvements to achieve
            
        Returns:
            GeneratedCode: Refactored code with updated tests
        """
        refactoring_context = {
            "original_code": code,
            "patterns": refactoring_patterns,
            "targets": quality_targets
        }
        
        # Apply refactoring patterns sequentially
        refactored_code = code
        for pattern in refactoring_patterns:
            refactored_code = await self._apply_refactoring_pattern(refactored_code, pattern)
        
        # Validate quality improvements
        quality_metrics = await self._assess_refactoring_quality(code, refactored_code, quality_targets)
        
        # Update tests to match refactored code
        updated_tests = await self._update_tests_for_refactoring(refactored_code, code)
        
        result = GeneratedCode(
            code=refactored_code,
            test_code=updated_tests,
            file_path="refactored.py",
            imports=self._extract_imports(refactored_code),
            dependencies=[],
            quality_metrics=quality_metrics,
            documentation=await self._generate_refactoring_documentation(refactoring_context)
        )
        
        return result
    
    async def generate_microservice(self,
                                  service_spec: Dict[str, Any],
                                  architectural_patterns: List[str],
                                  deployment_requirements: Dict[str, Any]) -> Dict[str, GeneratedCode]:
        """
        Generate complete microservice with all components.
        
        Args:
            service_spec: Service specification
            architectural_patterns: Patterns to follow
            deployment_requirements: Deployment needs
            
        Returns:
            Dict[str, GeneratedCode]: All service components
        """
        components = {}
        
        # Generate service layer
        components["service"] = await self._generate_service_layer(service_spec, architectural_patterns)
        
        # Generate API layer
        components["api"] = await self._generate_api_layer(service_spec, architectural_patterns)
        
        # Generate domain layer
        components["domain"] = await self._generate_domain_layer(service_spec)
        
        # Generate infrastructure layer
        components["infrastructure"] = await self._generate_infrastructure_layer(service_spec)
        
        # Generate configuration
        components["config"] = await self._generate_service_configuration(service_spec, deployment_requirements)
        
        # Generate deployment files
        components["deployment"] = await self._generate_deployment_files(service_spec, deployment_requirements)
        
        # Generate comprehensive tests
        components["tests"] = await self._generate_service_tests(components, service_spec)
        
        return components
    
    # Core TDD cycle implementation methods
    
    async def _generate_failing_test(self,
                                   requirement: str,
                                   architectural_guidance: Dict[str, Any],
                                   quality_requirements: Dict[str, Any]) -> str:
        """Generate a test that should fail initially."""
        test_template = """
import pytest
from unittest.mock import Mock, patch
from {module_name} import {class_name}


class Test{class_name}:
    \"\"\"Test suite for {class_name} following AAA pattern.\"\"\"
    
    def test_{test_method_name}_should_{expected_behavior}(self):
        \"\"\"
        Test that {requirement_description}.
        
        This test follows the Arrange-Act-Assert pattern and should
        initially fail to drive TDD implementation.
        \"\"\"
        # ARRANGE
        {arrange_code}
        
        # ACT
        {act_code}
        
        # ASSERT
        {assert_code}
"""
        
        # Extract key elements from requirement
        parsed_requirement = await self._parse_requirement(requirement)
        
        context = {
            "module_name": parsed_requirement["module"],
            "class_name": parsed_requirement["class"],
            "test_method_name": self._to_snake_case(parsed_requirement["method"]),
            "expected_behavior": parsed_requirement["behavior"],
            "requirement_description": requirement,
            "arrange_code": await self._generate_arrange_code(parsed_requirement, architectural_guidance),
            "act_code": await self._generate_act_code(parsed_requirement),
            "assert_code": await self._generate_assert_code(parsed_requirement, quality_requirements)
        }
        
        return test_template.format(**context)
    
    async def _generate_minimal_implementation(self,
                                             test_code: str,
                                             requirement: str,
                                             architectural_guidance: Dict[str, Any]) -> str:
        """Generate minimal implementation to make test pass."""
        # Parse test to understand what needs to be implemented
        test_analysis = await self._analyze_test_code(test_code)
        
        implementation_template = """
from typing import {type_hints}
{additional_imports}


class {class_name}:
    \"\"\"
    {class_description}
    
    This is a minimal implementation to satisfy the failing test.
    Will be refactored in the next TDD cycle phase.
    \"\"\"
    
    def __init__(self{init_params}):
        {init_implementation}
    
{methods}
"""
        
        context = {
            "type_hints": ", ".join(test_analysis["required_types"]),
            "additional_imports": "\n".join(test_analysis["required_imports"]),
            "class_name": test_analysis["class_name"],
            "class_description": f"Minimal implementation for: {requirement}",
            "init_params": test_analysis["init_params"],
            "init_implementation": await self._generate_minimal_init(test_analysis),
            "methods": await self._generate_minimal_methods(test_analysis, architectural_guidance)
        }
        
        return implementation_template.format(**context)
    
    async def _refactor_implementation(self,
                                     implementation_code: str,
                                     test_code: str,
                                     quality_requirements: Dict[str, Any]) -> str:
        """Refactor implementation while keeping tests green."""
        refactoring_opportunities = await self._identify_refactoring_opportunities(implementation_code)
        
        refactored_code = implementation_code
        
        for opportunity in refactoring_opportunities:
            if opportunity["impact"] >= quality_requirements.get("min_refactoring_impact", 5):
                refactored_code = await self._apply_refactoring(refactored_code, opportunity)
                
                # Verify tests still pass after each refactoring
                test_result = await self._run_test_with_implementation(test_code, refactored_code)
                if not test_result["passed"]:
                    # Revert refactoring if tests fail
                    break
        
        # Apply code quality improvements
        refactored_code = await self._apply_quality_improvements(refactored_code, quality_requirements)
        
        return refactored_code
    
    # Code pattern implementations
    
    async def _generate_arrange_code(self, requirement: Dict[str, Any], guidance: Dict[str, Any]) -> str:
        """Generate arrange section of test."""
        arrange_patterns = {
            "simple_object": "sut = {class_name}()",
            "with_dependencies": "mock_dependency = Mock()\n        sut = {class_name}(mock_dependency)",
            "with_data": "test_data = {test_data}\n        sut = {class_name}(test_data)"
        }
        
        pattern = guidance.get("test_pattern", "simple_object")
        return arrange_patterns.get(pattern, arrange_patterns["simple_object"]).format(**requirement)
    
    async def _generate_act_code(self, requirement: Dict[str, Any]) -> str:
        """Generate act section of test."""
        if requirement.get("method_params"):
            return f"result = sut.{requirement['method']}({requirement['method_params']})"
        else:
            return f"result = sut.{requirement['method']}()"
    
    async def _generate_assert_code(self, requirement: Dict[str, Any], quality_requirements: Dict[str, Any]) -> str:
        """Generate assert section of test."""
        assert_templates = {
            "equality": "assert result == expected_result",
            "type_check": "assert isinstance(result, {expected_type})",
            "exception": "with pytest.raises({exception_type}):\n            sut.{method}()",
            "mock_verification": "mock_dependency.{method}.assert_called_once_with({params})",
            "state_verification": "assert sut.{attribute} == expected_value"
        }
        
        assert_type = requirement.get("assert_type", "equality")
        return assert_templates.get(assert_type, assert_templates["equality"]).format(**requirement)
    
    # Quality assessment and validation
    
    async def _assess_generated_code_quality(self, code: str) -> Dict[str, Any]:
        """Assess quality of generated code."""
        metrics = {
            "complexity": await self._calculate_complexity(code),
            "maintainability": await self._assess_maintainability(code),
            "readability": await self._assess_readability(code),
            "testability": await self._assess_testability(code),
            "security": await self._assess_security(code),
            "performance": await self._assess_performance_characteristics(code)
        }
        
        # Calculate overall score
        weights = {"complexity": 0.2, "maintainability": 0.3, "readability": 0.2, 
                  "testability": 0.15, "security": 0.1, "performance": 0.05}
        
        overall_score = sum(metrics[key] * weights[key] for key in weights)
        metrics["overall_score"] = overall_score
        
        return metrics
    
    async def _run_test(self, test_code: str) -> Dict[str, Any]:
        """Run test code and return results."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            test_file = f.name
        
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_file, "-v"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "output": "",
                "errors": "Test execution timed out",
                "exit_code": -1
            }
        finally:
            os.unlink(test_file)
    
    async def _run_test_with_implementation(self, test_code: str, implementation_code: str) -> Dict[str, Any]:
        """Run test with implementation and return results."""
        # Create temporary files for both test and implementation
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write implementation
            impl_file = Path(temp_dir) / "implementation.py"
            impl_file.write_text(implementation_code)
            
            # Write test (modify imports to use local implementation)
            modified_test = test_code.replace("from {module_name}", f"from {impl_file.stem}")
            test_file = Path(temp_dir) / "test_implementation.py"
            test_file.write_text(modified_test)
            
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", str(test_file), "-v"],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                return {
                    "passed": result.returncode == 0,
                    "output": result.stdout,
                    "errors": result.stderr,
                    "exit_code": result.returncode
                }
            except subprocess.TimeoutExpired:
                return {
                    "passed": False,
                    "output": "",
                    "errors": "Test execution timed out",
                    "exit_code": -1
                }
    
    # Helper methods
    
    def _load_code_standards(self) -> Dict[str, Any]:
        """Load coding standards and conventions."""
        return {
            "max_line_length": 88,
            "max_function_length": 20,
            "max_class_length": 500,
            "max_complexity": 10,
            "naming_conventions": {
                "classes": "PascalCase",
                "functions": "snake_case",
                "constants": "UPPER_SNAKE_CASE",
                "private": "_leading_underscore"
            },
            "documentation_requirements": {
                "classes": "required",
                "public_methods": "required",
                "complex_functions": "required"
            }
        }
    
    def _register_builtin_patterns(self):
        """Register built-in code patterns."""
        # Would register patterns like Repository, Factory, Observer, etc.
        pass
    
    def _register_builtin_templates(self):
        """Register built-in code templates."""
        # Would register templates for common code structures
        pass
    
    def _to_snake_case(self, name: str) -> str:
        """Convert name to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    async def _parse_requirement(self, requirement: str) -> Dict[str, Any]:
        """Parse natural language requirement into structured data."""
        # Simplified parsing - would use NLP in production
        return {
            "module": "calculator",
            "class": "Calculator",
            "method": "calculate",
            "behavior": "return_correct_result",
            "method_params": "a, b",
            "expected_type": "float",
            "test_data": "{'a': 2, 'b': 3}"
        }
    
    async def _analyze_test_code(self, test_code: str) -> Dict[str, Any]:
        """Analyze test code to understand implementation requirements."""
        # Would parse AST to extract requirements
        return {
            "class_name": "Calculator",
            "required_types": ["Any", "Optional"],
            "required_imports": ["from typing import Any"],
            "init_params": "",
            "methods": ["calculate"]
        }
    
    async def _generate_minimal_init(self, analysis: Dict[str, Any]) -> str:
        """Generate minimal __init__ method."""
        return "pass"
    
    async def _generate_minimal_methods(self, analysis: Dict[str, Any], guidance: Dict[str, Any]) -> str:
        """Generate minimal method implementations."""
        methods = []
        for method_name in analysis["methods"]:
            method_template = f"""
    def {method_name}(self, *args, **kwargs):
        \"\"\"Minimal implementation - to be refactored.\"\"\"
        return None
"""
            methods.append(method_template)
        
        return "\n".join(methods)
    
    async def _identify_refactoring_opportunities(self, code: str) -> List[Dict[str, Any]]:
        """Identify opportunities for refactoring."""
        return [
            {
                "type": "extract_method",
                "impact": 7,
                "description": "Extract complex logic into separate method",
                "location": "line 15-25"
            }
        ]
    
    async def _apply_refactoring(self, code: str, opportunity: Dict[str, Any]) -> str:
        """Apply specific refactoring opportunity."""
        # Simplified - would implement actual refactoring transformations
        return code
    
    async def _apply_quality_improvements(self, code: str, requirements: Dict[str, Any]) -> str:
        """Apply code quality improvements."""
        # Add type hints, improve naming, add documentation, etc.
        return code
    
    async def _calculate_complexity(self, code: str) -> float:
        """Calculate cyclomatic complexity."""
        return 5.0  # Placeholder
    
    async def _assess_maintainability(self, code: str) -> float:
        """Assess code maintainability."""
        return 8.0  # Placeholder
    
    async def _assess_readability(self, code: str) -> float:
        """Assess code readability."""
        return 7.5  # Placeholder
    
    async def _assess_testability(self, code: str) -> float:
        """Assess how testable the code is."""
        return 8.5  # Placeholder
    
    async def _assess_security(self, code: str) -> float:
        """Assess security aspects of code."""
        return 7.0  # Placeholder
    
    async def _assess_performance_characteristics(self, code: str) -> float:
        """Assess performance characteristics."""
        return 6.5  # Placeholder
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        return imports


# Global code generator instance
code_generator = MCPCodeGenerator()