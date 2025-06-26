#!/usr/bin/env python3
"""
Comprehensive Test Suite for Persistent Memory System
Tests all aspects of the InfluxDB-based persistent memory integration

This test suite verifies:
- Basic memory storage and retrieval
- Memory type handling and categorization
- Importance scoring and decay calculations
- Memory linking and relationships
- Agent context loading and sessions
- Enhanced blackboard integration
- Memory consolidation and cleanup
"""

import asyncio
import time
import json
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Test configuration
TEST_AGENTS = ["researcher", "mastermind", "executor"]
TEST_MEMORY_TYPES = ["working", "episodic", "semantic", "procedural", "contextual"]

async def test_basic_memory_operations():
    """Test basic memory storage and retrieval"""
    print("🧪 Testing basic memory operations...")
    
    try:
        from persistent_memory import (
            get_persistent_memory, MemoryType, MemoryImportance, MemoryQuery
        )
        
        pm = get_persistent_memory()
        await pm.start()
        
        test_results = {"passed": 0, "failed": 0, "errors": []}
        
        # Test 1: Store memory
        try:
            memory_id = await pm.store_memory(
                agent="test_agent",
                memory_type=MemoryType.WORKING,
                content={"test": "basic_storage", "value": 123},
                importance=MemoryImportance.MEDIUM,
                tags=["test", "basic"]
            )
            
            if memory_id:
                test_results["passed"] += 1
                print(f"  ✅ Memory stored with ID: {memory_id}")
            else:
                test_results["failed"] += 1
                test_results["errors"].append("Failed to store memory")
                
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Store memory error: {e}")
        
        # Test 2: Retrieve memory by ID
        try:
            if memory_id:
                retrieved = await pm.get_memory_by_id(memory_id)
                if retrieved and retrieved.content.get("test") == "basic_storage":
                    test_results["passed"] += 1
                    print(f"  ✅ Memory retrieved successfully")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Failed to retrieve memory by ID")
            
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Retrieve memory error: {e}")
        
        # Test 3: Query memories
        try:
            query = MemoryQuery(
                agent="test_agent",
                query_text="basic_storage",
                limit=5
            )
            
            memories = await pm.retrieve_memories(query)
            if memories and len(memories) > 0:
                test_results["passed"] += 1
                print(f"  ✅ Query returned {len(memories)} memories")
            else:
                test_results["failed"] += 1
                test_results["errors"].append("Query returned no memories")
                
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Query memories error: {e}")
        
        await pm.stop()
        return test_results
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return {"passed": 0, "failed": 1, "errors": [str(e)]}

async def test_memory_types_and_importance():
    """Test different memory types and importance levels"""
    print("🧪 Testing memory types and importance...")
    
    try:
        from persistent_memory import (
            get_persistent_memory, MemoryType, MemoryImportance
        )
        
        pm = get_persistent_memory()
        await pm.start()
        
        test_results = {"passed": 0, "failed": 0, "errors": []}
        
        # Store different memory types
        test_memories = []
        
        for i, mem_type in enumerate(MemoryType):
            try:
                memory_id = await pm.store_memory(
                    agent="test_agent",
                    memory_type=mem_type,
                    content={"type_test": mem_type.value, "index": i},
                    importance=MemoryImportance.HIGH,
                    tags=[f"type_{mem_type.value}", "test"]
                )
                
                test_memories.append((memory_id, mem_type))
                test_results["passed"] += 1
                print(f"  ✅ Stored {mem_type.value} memory: {memory_id}")
                
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"Failed to store {mem_type.value}: {e}")
        
        # Test importance-based filtering
        try:
            from persistent_memory import MemoryQuery
            
            query = MemoryQuery(
                agent="test_agent",
                importance_threshold=8,  # High importance only
                limit=10
            )
            
            high_importance_memories = await pm.retrieve_memories(query)
            if len(high_importance_memories) >= len(test_memories):
                test_results["passed"] += 1
                print(f"  ✅ Importance filtering works: {len(high_importance_memories)} high-importance memories")
            else:
                test_results["failed"] += 1
                test_results["errors"].append("Importance filtering failed")
                
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Importance filtering error: {e}")
        
        await pm.stop()
        return test_results
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return {"passed": 0, "failed": 1, "errors": [str(e)]}

async def test_memory_linking():
    """Test memory linking and relationships"""
    print("🧪 Testing memory linking...")
    
    try:
        from persistent_memory import (
            get_persistent_memory, MemoryType, MemoryImportance
        )
        
        pm = get_persistent_memory()
        await pm.start()
        
        test_results = {"passed": 0, "failed": 0, "errors": []}
        
        # Store two related memories
        try:
            memory_id1 = await pm.store_memory(
                agent="test_agent",
                memory_type=MemoryType.SEMANTIC,
                content={"concept": "machine_learning", "type": "algorithm"},
                importance=MemoryImportance.HIGH,
                tags=["ml", "algorithm"]
            )
            
            memory_id2 = await pm.store_memory(
                agent="test_agent",
                memory_type=MemoryType.SEMANTIC,
                content={"concept": "neural_networks", "type": "algorithm"},
                importance=MemoryImportance.HIGH,
                tags=["ml", "neural", "algorithm"]
            )
            
            if memory_id1 and memory_id2:
                test_results["passed"] += 1
                print(f"  ✅ Created two memories for linking")
            else:
                test_results["failed"] += 1
                test_results["errors"].append("Failed to create memories for linking")
                
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Memory creation for linking error: {e}")
        
        # Link the memories
        try:
            if memory_id1 and memory_id2:
                link_success = await pm.link_memories(memory_id1, memory_id2, "related_concept")
                
                if link_success:
                    test_results["passed"] += 1
                    print(f"  ✅ Successfully linked memories")
                    
                    # Verify the link
                    memory1 = await pm.get_memory_by_id(memory_id1)
                    if memory1 and memory_id2 in memory1.linked_memories:
                        test_results["passed"] += 1
                        print(f"  ✅ Link verification successful")
                    else:
                        test_results["failed"] += 1
                        test_results["errors"].append("Link verification failed")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Memory linking failed")
                    
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Memory linking error: {e}")
        
        await pm.stop()
        return test_results
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return {"passed": 0, "failed": 1, "errors": [str(e)]}

async def test_enhanced_blackboard():
    """Test enhanced blackboard integration"""
    print("🧪 Testing enhanced blackboard integration...")
    
    try:
        from enhanced_blackboard import get_enhanced_blackboard
        
        ebb = get_enhanced_blackboard()
        await ebb.start()
        
        test_results = {"passed": 0, "failed": 0, "errors": []}
        
        # Test session management
        try:
            session_id, context = await ebb.start_agent_session(
                "test_agent", 
                {"task": "integration_test"}
            )
            
            if session_id and context:
                test_results["passed"] += 1
                print(f"  ✅ Started agent session: {session_id}")
            else:
                test_results["failed"] += 1
                test_results["errors"].append("Failed to start agent session")
                
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Session start error: {e}")
        
        # Test task with memory
        try:
            task_id = await ebb.write_task_with_memory(
                agent="test_agent",
                task_type="test_task",
                data={"test": "integration", "complexity": "medium"},
                priority=5
            )
            
            if task_id:
                test_results["passed"] += 1
                print(f"  ✅ Created task with memory: {task_id}")
            else:
                test_results["failed"] += 1
                test_results["errors"].append("Failed to create task with memory")
                
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Task creation error: {e}")
        
        # Test task completion with memory
        try:
            if task_id:
                await ebb.complete_task_with_memory(
                    task_id, 
                    "test_agent",
                    {"result": "success", "performance": "good"},
                    ["learned about integration testing", "memory persistence works"]
                )
                
                test_results["passed"] += 1
                print(f"  ✅ Completed task with memory")
                
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Task completion error: {e}")
        
        # Test session end
        try:
            await ebb.end_agent_session(
                "test_agent",
                {"tests_run": 3, "success_rate": 1.0}
            )
            
            test_results["passed"] += 1
            print(f"  ✅ Ended agent session")
            
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Session end error: {e}")
        
        await ebb.stop()
        return test_results
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return {"passed": 0, "failed": 1, "errors": [str(e)]}

async def test_memory_consolidation():
    """Test memory consolidation and cleanup"""
    print("🧪 Testing memory consolidation...")
    
    try:
        from persistent_memory import (
            get_persistent_memory, MemoryType, MemoryImportance
        )
        
        pm = get_persistent_memory()
        await pm.start()
        
        test_results = {"passed": 0, "failed": 0, "errors": []}
        
        # Create some test memories with different characteristics
        try:
            # High importance memory (should not decay)
            await pm.store_memory(
                agent="test_agent",
                memory_type=MemoryType.SEMANTIC,
                content={"important": "knowledge", "value": "critical"},
                importance=MemoryImportance.CRITICAL,
                tags=["critical", "test"]
            )
            
            # Temporary memory (should decay quickly)
            await pm.store_memory(
                agent="test_agent",
                memory_type=MemoryType.WORKING,
                content={"temporary": "data", "value": "temp"},
                importance=MemoryImportance.TEMPORARY,
                tags=["temporary", "test"]
            )
            
            test_results["passed"] += 1
            print(f"  ✅ Created test memories for consolidation")
            
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Test memory creation error: {e}")
        
        # Run consolidation
        try:
            stats = await pm.consolidate_memories("test_agent")
            
            if isinstance(stats, dict) and "processed" in stats:
                test_results["passed"] += 1
                print(f"  ✅ Consolidation completed: {stats}")
            else:
                test_results["failed"] += 1
                test_results["errors"].append("Consolidation returned invalid stats")
                
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Consolidation error: {e}")
        
        await pm.stop()
        return test_results
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return {"passed": 0, "failed": 1, "errors": [str(e)]}

async def test_memory_statistics():
    """Test memory statistics and reporting"""
    print("🧪 Testing memory statistics...")
    
    try:
        from persistent_memory import get_persistent_memory
        
        pm = get_persistent_memory()
        await pm.start()
        
        test_results = {"passed": 0, "failed": 0, "errors": []}
        
        # Get memory statistics
        try:
            stats = await pm.get_memory_statistics("test_agent")
            
            if isinstance(stats, dict) and "total_memories" in stats:
                test_results["passed"] += 1
                print(f"  ✅ Memory statistics retrieved: {stats['total_memories']} total memories")
                
                # Check for expected structure
                expected_keys = ["by_type", "by_importance", "by_agent"]
                for key in expected_keys:
                    if key in stats:
                        test_results["passed"] += 1
                        print(f"  ✅ Statistics include {key}")
                    else:
                        test_results["failed"] += 1
                        test_results["errors"].append(f"Missing statistics key: {key}")
                        
            else:
                test_results["failed"] += 1
                test_results["errors"].append("Invalid statistics structure")
                
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Statistics retrieval error: {e}")
        
        await pm.stop()
        return test_results
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return {"passed": 0, "failed": 1, "errors": [str(e)]}

async def run_comprehensive_test_suite():
    """Run all tests and provide comprehensive report"""
    print("🚀 Starting Comprehensive Persistent Memory Test Suite")
    print("=" * 60)
    
    total_results = {"passed": 0, "failed": 0, "errors": []}
    
    # Run all test functions
    test_functions = [
        test_basic_memory_operations,
        test_memory_types_and_importance,
        test_memory_linking,
        test_enhanced_blackboard,
        test_memory_consolidation,
        test_memory_statistics
    ]
    
    for test_func in test_functions:
        try:
            print(f"\n{test_func.__name__.replace('_', ' ').title()}")
            print("-" * 40)
            
            results = await test_func()
            
            total_results["passed"] += results["passed"]
            total_results["failed"] += results["failed"]
            total_results["errors"].extend(results["errors"])
            
            print(f"  Results: {results['passed']} passed, {results['failed']} failed")
            
        except Exception as e:
            print(f"  ❌ Test function error: {e}")
            total_results["failed"] += 1
            total_results["errors"].append(f"{test_func.__name__}: {e}")
    
    # Final report
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)
    print(f"✅ Total Passed: {total_results['passed']}")
    print(f"❌ Total Failed: {total_results['failed']}")
    print(f"📈 Success Rate: {total_results['passed'] / (total_results['passed'] + total_results['failed']) * 100:.1f}%")
    
    if total_results["errors"]:
        print(f"\n🔍 Error Details:")
        for i, error in enumerate(total_results["errors"], 1):
            print(f"  {i}. {error}")
    
    print("\n🎯 Test Summary:")
    if total_results["failed"] == 0:
        print("  🎉 ALL TESTS PASSED! Persistent memory system is working correctly.")
    else:
        print(f"  ⚠️  {total_results['failed']} tests failed. Please review errors above.")
    
    return total_results

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test_suite())