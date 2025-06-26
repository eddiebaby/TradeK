#!/usr/bin/env python3
"""
Interactive script to give tasks to MASTERMIND agent
Now with InfluxDB blackboard integration for trio collaboration
"""
import asyncio
import sys
import time
import json
from pathlib import Path

# Add agents directory to path
sys.path.append(str(Path(__file__).parent))

from mastermind.mastermind_agent import MastermindAgent
from influx_blackboard import write_task, read_tasks, update_status, log_performance, write_reflection, get_data

async def process_strategic_task(mastermind, task_id, task_data):
    """Process a strategic analysis task from the blackboard"""
    try:
        print(f"🔍 Processing strategic task: {task_id}")
        start_time = time.time()
        
        # Extract task information
        research_topic = task_data.get('research_topic', 'Unknown')
        research_results = task_data.get('research_results', {})
        confidence_level = task_data.get('confidence_level', 0.5)
        
        print(f"📊 Research confidence: {confidence_level:.2f}")
        print(f"🎯 Topic: {research_topic}")
        
        # Update task status
        await update_status(task_id, "in_progress", "MASTERMIND")
        
        # Perform strategic analysis
        result = await mastermind.strategic_analysis(research_topic, context=research_results)
        
        # Calculate performance
        execution_time = time.time() - start_time
        tokens_used = len(str(result)) // 4
        
        # Log performance
        await log_performance("MASTERMIND", "blackboard_strategic_analysis", tokens_used, execution_time, True, len(str(result)))
        
        # Mark task as completed
        await update_status(task_id, "completed", "MASTERMIND")
        
        # Write reflection
        quality_score = 0.8  # Base quality score
        await write_reflection(
            "MASTERMIND",
            "collaboration",
            "medium", 
            f"Completed strategic analysis for research task {task_id}",
            "Continue processing blackboard tasks efficiently",
            quality_score
        )
        
        # Check if we should hand off to EXECUTOR
        if "executor_recommendations" in result and quality_score > 0.7:
            print("🤝 Strategic analysis complete - preparing handoff to EXECUTOR...")
            executor_task_id = await write_task("EXECUTOR", "implementation_planning", {
                "strategic_analysis": result,
                "source_task": task_id,
                "research_topic": research_topic,
                "confidence_level": quality_score,
                "handoff_reason": "strategic_analysis_complete"
            }, priority=1)
            print(f"📤 EXECUTOR task created: {executor_task_id}")
        
        print(f"✅ Strategic task {task_id} completed successfully")
        
    except Exception as e:
        print(f"❌ Error processing task {task_id}: {e}")
        await update_status(task_id, "failed", "MASTERMIND")
        await write_reflection(
            "MASTERMIND",
            "error",
            "high",
            f"Failed to process task {task_id}: {str(e)}",
            "Review task processing pipeline",
            0.1
        )

async def main():
    print("🧠 MASTERMIND Strategic Agent")
    print("=" * 50)
    
    mastermind = MastermindAgent()
    
    # Check for pending tasks from blackboard first
    print("🔍 Checking blackboard for pending strategic analysis tasks...")
    pending_tasks = await read_tasks("MASTERMIND", "new")
    
    if pending_tasks:
        print(f"📋 Found {len(pending_tasks)} pending tasks from blackboard")
        for task_data in pending_tasks[:3]:  # Process up to 3 tasks
            task_id = task_data.get('id')
            task_type = task_data.get('type')
            print(f"📤 Processing blackboard task: {task_id} ({task_type})")
            
            # Get full task data
            full_data = await get_data(task_id)
            if full_data:
                await process_strategic_task(mastermind, task_id, full_data)
    
    while True:
        print("\n" + "="*60)
        print("What would you like MASTERMIND to analyze?")
        print("(Type 'quit' to exit, 'check' to check blackboard)")
        
        task = input("📝 Enter your task: ").strip()
        
        if task.lower() in ['quit', 'exit', 'q']:
            break
            
        if task.lower() == 'check':
            # Check blackboard for new tasks
            new_tasks = await read_tasks("MASTERMIND", "new", 5)
            if new_tasks:
                print(f"📋 Found {len(new_tasks)} new tasks:")
                for i, task_data in enumerate(new_tasks, 1):
                    print(f"   {i}. {task_data.get('id')} - {task_data.get('type')}")
                    # Process each task
                    full_data = await get_data(task_data.get('id'))
                    if full_data:
                        await process_strategic_task(mastermind, task_data.get('id'), full_data)
            else:
                print("📭 No new tasks found on blackboard")
            continue
            
        if not task:
            continue
            
        print(f"\n🧠 MASTERMIND analyzing: '{task}'")
        print("⏳ Thinking strategically...")
        
        try:
            start_time = time.time()
            
            # Log task start
            await log_performance("MASTERMIND", "strategic_start", 30, 0.1, True)
            
            result = await mastermind.strategic_analysis(task)
            
            # Calculate performance
            execution_time = time.time() - start_time
            tokens_used = len(str(result)) // 4
            
            # Log performance
            await log_performance("MASTERMIND", "strategic_analysis", tokens_used, execution_time, True, len(str(result)))
            
            print("\n✅ MASTERMIND Analysis Complete!")
            print("-" * 40)
            
            # Show key results
            if "architecture_design" in result:
                arch = result["architecture_design"]
                print(f"🏗️  Architecture: {arch.get('architectural_style', 'N/A')}")
                
            if "quality_strategy" in result:
                strategy = result["quality_strategy"]
                print(f"📊 Quality Strategy: {strategy.strategy_id}")
                
            if "risk_assessment" in result:
                risks = result["risk_assessment"]
                total_risks = sum(len(risks.get(category, [])) for category in risks)
                print(f"⚠️  Risk Assessment: {total_risks} risks identified")
                
            if "executor_recommendations" in result:
                recs = result["executor_recommendations"]
                methodology = recs.get("implementation_approach", {}).get("methodology", "N/A")
                print(f"🛠️  Implementation Approach: {methodology}")
                
            print("-" * 40)
            
            # Write reflection on strategic analysis
            await write_reflection(
                "MASTERMIND",
                "performance",
                "medium",
                f"Completed user-requested strategic analysis for '{task}'",
                "Continue delivering high-quality strategic analysis",
                0.8
            )
            
            # Check if we should create implementation tasks for EXECUTOR
            if "executor_recommendations" in result:
                executor_recs = result["executor_recommendations"]
                if executor_recs.get("implementation_approach"):
                    print("🤝 Strategy complete - preparing implementation tasks for EXECUTOR...")
                    executor_task_id = await write_task("EXECUTOR", "user_implementation_request", {
                        "strategic_analysis": result,
                        "user_task": task,
                        "implementation_approach": executor_recs,
                        "handoff_reason": "user_task_implementation"
                    }, priority=1)
                    print(f"📤 EXECUTOR task created: {executor_task_id}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            # Log error to blackboard
            await log_performance("MASTERMIND", "strategic_error", 0, 0, False)
            await write_reflection(
                "MASTERMIND",
                "error",
                "high", 
                f"Strategic analysis failed for '{task}': {str(e)}",
                "Review error patterns and improve error handling",
                0.2
            )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Strategic session ended.")
    except Exception as e:
        print(f"❌ Error: {e}")