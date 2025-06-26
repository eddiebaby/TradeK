#!/usr/bin/env python3
"""
Interactive script to give implementation tasks to EXECUTOR agent
Now with InfluxDB blackboard integration for trio collaboration
"""
import asyncio
import sys
import json
import time
from pathlib import Path

# Add agents directory to path
sys.path.append(str(Path(__file__).parent))

from executor.executor_agent import ExecutorAgent
from core.agent_base import TaskContext
from influx_blackboard import write_task, read_tasks, update_status, log_performance, write_reflection, get_data

async def process_implementation_task(executor, task_id, task_data):
    """Process an implementation task from the blackboard"""
    try:
        print(f"🔧 Processing implementation task: {task_id}")
        start_time = time.time()
        
        # Extract task information
        strategic_analysis = task_data.get('strategic_analysis', {})
        research_topic = task_data.get('research_topic', 'Unknown')
        implementation_approach = task_data.get('implementation_approach', {})
        user_task = task_data.get('user_task', research_topic)
        
        print(f"🎯 Implementation target: {research_topic}")
        print(f"📋 User request: {user_task}")
        
        # Update task status
        await update_status(task_id, "in_progress", "EXECUTOR")
        
        # Create implementation task context
        task_context = executor.create_task_context(
            description=user_task,
            requirements={
                "implementation_type": "tdd",
                "quality_gates": True,
                "test_coverage": 95,
                "based_on_strategy": True
            },
            constraints={
                "strategic_guidance": strategic_analysis,
                "quality": "production_ready"
            },
            performance_targets={
                "response_time": "< 100ms",
                "throughput": "> 1000 rps"
            },
            success_criteria={
                "all_tests_pass": True,
                "strategy_implemented": True,
                "quality_gates_pass": True
            }
        )
        
        # Execute implementation
        result = await executor.process_task(task_context)
        
        # Calculate performance
        execution_time = time.time() - start_time
        tokens_used = len(str(result)) // 4
        
        # Log performance
        await log_performance("EXECUTOR", "blackboard_implementation", tokens_used, execution_time, True, len(str(result)))
        
        # Mark task as completed
        await update_status(task_id, "completed", "EXECUTOR")
        
        # Write reflection on implementation quality
        quality_score = result.get("quality_report", {}).get("overall_score", 0.7)
        await write_reflection(
            "EXECUTOR",
            "collaboration",
            "medium",
            f"Completed implementation for strategic task {task_id} with quality {quality_score:.2f}",
            "Continue delivering high-quality implementations from strategic guidance",
            quality_score
        )
        
        # Log implementation completion back to blackboard for monitoring
        await write_task("EXECUTOR", "implementation_completed", {
            "completed_task": task_id,
            "quality_score": quality_score,
            "implementation_summary": result.get("execution_summary", {}),
            "completion_time": execution_time,
            "trio_workflow": "complete"
        }, priority=3)
        
        print(f"✅ Implementation task {task_id} completed successfully")
        
    except Exception as e:
        print(f"❌ Error processing implementation task {task_id}: {e}")
        await update_status(task_id, "failed", "EXECUTOR")
        await write_reflection(
            "EXECUTOR",
            "error",
            "high",
            f"Failed to process implementation task {task_id}: {str(e)}",
            "Review implementation processing pipeline",
            0.1
        )

async def main():
    print("⚡ EXECUTOR Implementation Agent")
    print("=" * 50)
    
    executor = ExecutorAgent()
    
    # Check for pending implementation tasks from blackboard first
    print("🔍 Checking blackboard for pending implementation tasks...")
    pending_tasks = await read_tasks("EXECUTOR", "new")
    
    if pending_tasks:
        print(f"📋 Found {len(pending_tasks)} pending tasks from blackboard")
        for task_data in pending_tasks[:3]:  # Process up to 3 tasks
            task_id = task_data.get('id')
            task_type = task_data.get('type')
            print(f"📤 Processing blackboard task: {task_id} ({task_type})")
            
            # Get full task data
            full_data = await get_data(task_id)
            if full_data:
                await process_implementation_task(executor, task_id, full_data)
    
    print("🛠️  Implementation Capabilities:")
    capabilities = executor.get_capabilities()
    for i, capability in enumerate(capabilities[:6], 1):
        print(f"   {i}. {capability.replace('_', ' ').title()}")
    
    print("\n⚙️  Execution Modes:")
    modes = executor.get_thinking_modes()
    for mode, description in list(modes.items())[:4]:
        print(f"   • {mode}: {description}")
    
    while True:
        print("\n" + "="*60)
        print("What would you like EXECUTOR to implement?")
        print("(Type 'quit' to exit, 'help' for examples, 'check' to check blackboard)")
        
        task = input("⚡ Enter implementation task: ").strip()
        
        if task.lower() in ['quit', 'exit', 'q']:
            break
            
        if task.lower() == 'help':
            print("\n📝 Example Implementation Tasks:")
            print("   • 'implement a REST API endpoint for user authentication'")
            print("   • 'create a caching layer with Redis integration'")
            print("   • 'build a real-time data processing pipeline'")
            print("   • 'implement TDD workflow for trading algorithm'")
            print("   • 'create comprehensive test suite for search API'")
            print("   • 'setup CI/CD pipeline with quality gates'")
            continue
            
        if task.lower() == 'check':
            # Check blackboard for new implementation tasks
            new_tasks = await read_tasks("EXECUTOR", "new", 5)
            if new_tasks:
                print(f"📋 Found {len(new_tasks)} new implementation tasks:")
                for i, task_data in enumerate(new_tasks, 1):
                    print(f"   {i}. {task_data.get('id')} - {task_data.get('type')}")
                    # Process each task
                    full_data = await get_data(task_data.get('id'))
                    if full_data:
                        await process_implementation_task(executor, task_data.get('id'), full_data)
            else:
                print("📭 No new implementation tasks found on blackboard")
            continue
            
        if not task:
            continue
            
        print(f"\n⚡ EXECUTOR implementing: '{task}'")
        print("🔄 Following TDD Red-Green-Refactor cycle...")
        
        try:
            start_time = time.time()
            
            # Log implementation start
            await log_performance("EXECUTOR", "implementation_start", 40, 0.1, True)
            
            # Create task context
            task_context = executor.create_task_context(
                description=task,
                requirements={
                    "implementation_type": "tdd",
                    "quality_gates": True,
                    "test_coverage": 95,
                    "mutation_score": 85
                },
                constraints={
                    "time_limit": "reasonable",
                    "code_quality": "production_ready"
                },
                performance_targets={
                    "response_time": "< 100ms",
                    "throughput": "> 1000 rps"
                },
                success_criteria={
                    "all_tests_pass": True,
                    "coverage_target_met": True,
                    "quality_gates_pass": True
                }
            )
            
            # Execute implementation
            result = await executor.process_task(task_context)
            
            # Calculate performance
            execution_time = time.time() - start_time
            tokens_used = len(str(result)) // 4
            
            # Log performance
            await log_performance("EXECUTOR", "user_implementation", tokens_used, execution_time, True, len(str(result)))
            
            print("\n✅ EXECUTOR Implementation Complete!")
            print("-" * 50)
            
            # Show implementation results
            impl_result = result.get("implementation_result", {})
            if impl_result:
                print(f"📦 Implementation:")
                print(f"   • Status: {impl_result.get('status', 'N/A')}")
                print(f"   • Code Files: {len(impl_result.get('code_files', []))}")
                print(f"   • Test Files: {len(impl_result.get('test_files', []))}")
            
            # Show test suite results
            test_suite = result.get("test_suite", {})
            if test_suite:
                print(f"\n🧪 Test Suite:")
                print(f"   • Unit Tests: {len(test_suite.get('unit_tests', []))}")
                print(f"   • Integration Tests: {len(test_suite.get('integration_tests', []))}")
                print(f"   • Property Tests: {len(test_suite.get('property_tests', []))}")
                print(f"   • Mutation Tests: {len(test_suite.get('mutation_tests', []))}")
                
                coverage = test_suite.get('coverage_report', {})
                if coverage:
                    print(f"   • Coverage: {coverage.get('percentage', 0)}%")
            
            # Show quality metrics
            quality_report = result.get("quality_report", {})
            if quality_report:
                print(f"\n📊 Quality Metrics:")
                print(f"   • Overall Score: {quality_report.get('overall_score', 0):.2f}")
                print(f"   • Code Coverage: {quality_report.get('coverage', 0)}%")
                print(f"   • Mutation Score: {quality_report.get('mutation_score', 0)}%")
                print(f"   • Security Score: {quality_report.get('security_score', 0):.2f}")
                print(f"   • Performance Score: {quality_report.get('performance_score', 0):.2f}")
            
            # Show monitoring setup
            monitoring = result.get("monitoring_setup", {})
            if monitoring:
                print(f"\n📈 Monitoring & Observability:")
                metrics = monitoring.get('metrics', [])
                if metrics:
                    print(f"   • Metrics: {', '.join(metrics[:3])}")
                
                health_checks = monitoring.get('health_checks', [])
                if health_checks:
                    print(f"   • Health Checks: {len(health_checks)} configured")
            
            # Show deployment pipeline
            deployment = result.get("deployment_pipeline", {})
            if deployment:
                print(f"\n🚀 Deployment Pipeline:")
                stages = deployment.get('stages', [])
                if stages:
                    print(f"   • Pipeline Stages: {len(stages)}")
                    for stage in stages[:3]:
                        if isinstance(stage, dict):
                            print(f"     - {stage.get('name', 'Unknown')}")
                        else:
                            print(f"     - {stage}")
                
                quality_gates = deployment.get('quality_gates', [])
                if quality_gates:
                    print(f"   • Quality Gates: {len(quality_gates)} configured")
            
            # Show execution summary
            summary = result.get("execution_summary", {})
            if summary:
                print(f"\n📋 Execution Summary:")
                print(f"   • Implementation Approach: {summary.get('approach', 'N/A')}")
                print(f"   • Quality Level: {summary.get('quality_level', 'N/A')}")
                print(f"   • Performance Tier: {summary.get('performance_tier', 'N/A')}")
                print(f"   • Security Level: {summary.get('security_level', 'N/A')}")
                
                recommendations = summary.get('recommendations', [])
                if recommendations:
                    print(f"   • Next Steps: {recommendations[0]}")
            
            print("-" * 50)
            
            # Write reflection on implementation quality
            quality_score = result.get("quality_report", {}).get("overall_score", 0.7)
            await write_reflection(
                "EXECUTOR",
                "performance",
                "medium",
                f"Completed user implementation for '{task}' with quality {quality_score:.2f}",
                "Continue delivering high-quality TDD implementations",
                quality_score
            )
            
            # Create monitoring task for ongoing quality tracking
            monitoring_task_id = await write_task("EXECUTOR", "implementation_monitoring", {
                "implementation_task": task,
                "quality_metrics": result.get("quality_report", {}),
                "monitoring_setup": result.get("monitoring_setup", {}),
                "deployment_pipeline": result.get("deployment_pipeline", {}),
                "monitoring_reason": "track_implementation_performance"
            }, priority=3)
            print(f"📊 Monitoring task created: {monitoring_task_id}")
            
        except Exception as e:
            print(f"❌ Implementation Error: {e}")
            # Log error to blackboard
            await log_performance("EXECUTOR", "implementation_error", 0, 0, False)
            await write_reflection(
                "EXECUTOR",
                "error",
                "high",
                f"Implementation failed for '{task}': {str(e)}",
                "Review implementation error patterns and improve resilience",
                0.2
            )
            import traceback
            traceback.print_exc()

async def quick_demo():
    """Quick demo of EXECUTOR capabilities"""
    print("🚀 EXECUTOR Quick Demo")
    print("=" * 40)
    
    executor = ExecutorAgent()
    
    # Demo task
    demo_task = "Create a simple REST API endpoint with authentication"
    
    print(f"⚡ Demo Task: {demo_task}")
    print("🔄 Running TDD implementation...")
    
    task_context = executor.create_task_context(
        description=demo_task,
        requirements={"demo": True, "quick": True},
        constraints={},
        performance_targets={},
        success_criteria={"demo_complete": True}
    )
    
    result = await executor.process_task(task_context)
    
    print("✅ Demo Complete!")
    print(f"📦 Generated {len(result.get('implementation_result', {}).get('code_files', []))} code files")
    print(f"🧪 Created {len(result.get('test_suite', {}).get('unit_tests', []))} test files")
    print(f"📊 Quality Score: {result.get('quality_report', {}).get('overall_score', 0):.2f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        try:
            asyncio.run(quick_demo())
        except Exception as e:
            print(f"❌ Demo Error: {e}")
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n👋 Implementation session ended.")
        except Exception as e:
            print(f"❌ Error: {e}")