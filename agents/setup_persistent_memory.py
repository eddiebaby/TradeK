#!/usr/bin/env python3
"""
Setup Script for Persistent Memory System
Initializes InfluxDB buckets and validates the persistent memory system

This script:
- Checks InfluxDB connectivity and configuration
- Creates necessary buckets for memory storage
- Validates the persistent memory system
- Sets up initial agent memory contexts
- Provides system health checks
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    dependencies = {
        "influxdb_client": "pip install influxdb-client",
        "yaml": "pip install pyyaml",
        "numpy": "pip install numpy"
    }
    
    missing = []
    for dep, install_cmd in dependencies.items():
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} - Run: {install_cmd}")
            missing.append(dep)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        return False
    
    print("✅ All dependencies available")
    return True

def check_influxdb_config():
    """Check InfluxDB configuration"""
    print("\n🔍 Checking InfluxDB configuration...")
    
    config_path = Path(__file__).parent / "config" / "blackboard_influx.yaml"
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ["influxdb", "token_optimization", "monitoring"]
        for key in required_keys:
            if key in config:
                print(f"  ✅ {key} configuration found")
            else:
                print(f"  ❌ Missing {key} configuration")
                return False
        
        # Check token file
        token_file = Path(config["influxdb"]["token_file"])
        if token_file.exists():
            print(f"  ✅ InfluxDB token file found")
        else:
            print(f"  ⚠️  InfluxDB token file not found: {token_file}")
            print(f"     You may need to run the InfluxDB setup script first")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False

async def check_influxdb_connection():
    """Check InfluxDB connection"""
    print("\n🔍 Checking InfluxDB connection...")
    
    try:
        from influx_blackboard import get_blackboard
        
        blackboard = get_blackboard()
        
        if blackboard.client:
            print("  ✅ InfluxDB client initialized")
            
            # Test connection
            try:
                health = blackboard.client.health()
                if health.status == "pass":
                    print("  ✅ InfluxDB connection healthy")
                    return True
                else:
                    print(f"  ❌ InfluxDB health check failed: {health.message}")
                    return False
            except Exception as e:
                print(f"  ❌ InfluxDB health check error: {e}")
                return False
        else:
            print("  ❌ InfluxDB client not initialized")
            return False
            
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Connection error: {e}")
        return False

async def create_memory_buckets():
    """Create necessary InfluxDB buckets for memory storage"""
    print("\n🏗️  Creating memory buckets...")
    
    try:
        from influx_blackboard import get_blackboard
        
        blackboard = get_blackboard()
        
        if not blackboard.client:
            print("  ❌ No InfluxDB client available")
            return False
        
        # Memory buckets to create
        memory_buckets = [
            "working_memory",
            "episodic_memory", 
            "semantic_memory",
            "procedural_memory",
            "contextual_memory",
            "relationships",
            "archive_working_memory",
            "archive_episodic_memory",
            "archive_semantic_memory",
            "archive_contextual_memory",
            "expired"
        ]
        
        buckets_api = blackboard.client.buckets_api()
        existing_buckets = buckets_api.find_buckets()
        existing_names = [bucket.name for bucket in existing_buckets.buckets or []]
        
        created_count = 0
        for bucket_name in memory_buckets:
            if bucket_name not in existing_names:
                try:
                    # Create bucket with appropriate retention
                    retention_days = 30
                    if "archive" in bucket_name:
                        retention_days = 365
                    elif "expired" in bucket_name:
                        retention_days = 7
                    elif "procedural" in bucket_name:
                        retention_days = 0  # No expiration for procedural memory
                    
                    retention_rules = []
                    if retention_days > 0:
                        retention_rules.append({
                            "type": "expire",
                            "everySeconds": retention_days * 24 * 3600
                        })
                    
                    buckets_api.create_bucket(
                        bucket_name=bucket_name,
                        org=blackboard.config["influxdb"]["org"],
                        retention_rules=retention_rules
                    )
                    
                    print(f"  ✅ Created bucket: {bucket_name} (retention: {retention_days} days)")
                    created_count += 1
                    
                except Exception as e:
                    print(f"  ❌ Failed to create bucket {bucket_name}: {e}")
            else:
                print(f"  ℹ️  Bucket already exists: {bucket_name}")
        
        print(f"✅ Memory buckets setup complete. Created {created_count} new buckets.")
        return True
        
    except Exception as e:
        print(f"❌ Error creating memory buckets: {e}")
        return False

async def validate_persistent_memory():
    """Validate persistent memory system functionality"""
    print("\n🧪 Validating persistent memory system...")
    
    try:
        from persistent_memory import get_persistent_memory, MemoryType, MemoryImportance
        
        pm = get_persistent_memory()
        await pm.start()
        
        # Test basic functionality
        print("  🔬 Testing basic memory operations...")
        
        # Store test memory
        memory_id = await pm.store_memory(
            agent="setup_test",
            memory_type=MemoryType.WORKING,
            content={"test": "setup_validation", "timestamp": time.time()},
            importance=MemoryImportance.MEDIUM,
            tags=["setup", "validation"]
        )
        
        if memory_id:
            print(f"  ✅ Successfully stored test memory: {memory_id}")
        else:
            print("  ❌ Failed to store test memory")
            return False
        
        # Retrieve test memory
        retrieved = await pm.get_memory_by_id(memory_id)
        
        if retrieved and retrieved.content.get("test") == "setup_validation":
            print("  ✅ Successfully retrieved test memory")
        else:
            print("  ❌ Failed to retrieve test memory")
            return False
        
        # Test query functionality
        from persistent_memory import MemoryQuery
        
        query = MemoryQuery(
            agent="setup_test",
            tags=["setup"],
            limit=5
        )
        
        memories = await pm.retrieve_memories(query)
        
        if memories and len(memories) > 0:
            print(f"  ✅ Query returned {len(memories)} memories")
        else:
            print("  ❌ Query returned no memories")
            return False
        
        await pm.stop()
        print("✅ Persistent memory system validation successful")
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

async def validate_enhanced_blackboard():
    """Validate enhanced blackboard integration"""
    print("\n🧪 Validating enhanced blackboard integration...")
    
    try:
        from enhanced_blackboard import get_enhanced_blackboard
        
        ebb = get_enhanced_blackboard()
        await ebb.start()
        
        # Test session management
        session_id, context = await ebb.start_agent_session(
            "setup_test",
            {"validation": True}
        )
        
        if session_id:
            print("  ✅ Agent session started successfully")
        else:
            print("  ❌ Failed to start agent session")
            return False
        
        # Test task with memory
        task_id = await ebb.write_task_with_memory(
            agent="setup_test",
            task_type="validation_task",
            data={"validation": True, "setup": "test"},
            priority=5
        )
        
        if task_id:
            print("  ✅ Task with memory created successfully")
        else:
            print("  ❌ Failed to create task with memory")
            return False
        
        # Complete task
        await ebb.complete_task_with_memory(
            task_id,
            "setup_test",
            {"result": "validation_successful"},
            ["Setup validation completed"]
        )
        print("  ✅ Task completed with memory")
        
        # End session
        await ebb.end_agent_session(
            "setup_test",
            {"validation": "completed"}
        )
        print("  ✅ Agent session ended successfully")
        
        await ebb.stop()
        print("✅ Enhanced blackboard validation successful")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced blackboard validation error: {e}")
        return False

async def setup_agent_contexts():
    """Setup initial contexts for known agents"""
    print("\n👥 Setting up agent contexts...")
    
    try:
        from persistent_memory import get_persistent_memory, MemoryType, MemoryImportance
        
        pm = get_persistent_memory()
        await pm.start()
        
        # Agent configurations
        agents = {
            "researcher": {
                "role": "Knowledge Architect & Intelligence Synthesizer",
                "capabilities": ["multi-source intelligence", "security research", "performance benchmarking"],
                "memory_profile": "high_semantic_storage"
            },
            "mastermind": {
                "role": "Strategic Architect & Quality Orchestrator",
                "capabilities": ["strategic analysis", "architectural design", "quality orchestration"],
                "memory_profile": "high_procedural_storage"
            },
            "executor": {
                "role": "Implementation Virtuoso & Operational Expert",
                "capabilities": ["TDD implementation", "comprehensive testing", "DevOps automation"],
                "memory_profile": "high_episodic_storage"
            }
        }
        
        for agent_name, config in agents.items():
            try:
                # Store agent profile in procedural memory
                await pm.store_memory(
                    agent=agent_name,
                    memory_type=MemoryType.PROCEDURAL,
                    content=config,
                    importance=MemoryImportance.CRITICAL,
                    tags=["agent_profile", "configuration", "persistent"]
                )
                
                # Store initial context
                await pm.store_memory(
                    agent=agent_name,
                    memory_type=MemoryType.CONTEXTUAL,
                    content={
                        "initialization": "system_setup",
                        "setup_time": time.time(),
                        "status": "ready"
                    },
                    importance=MemoryImportance.HIGH,
                    tags=["initialization", "context", "setup"]
                )
                
                print(f"  ✅ {agent_name} context initialized")
                
            except Exception as e:
                print(f"  ❌ Failed to initialize {agent_name}: {e}")
        
        await pm.stop()
        print("✅ Agent contexts setup complete")
        return True
        
    except Exception as e:
        print(f"❌ Agent context setup error: {e}")
        return False

async def generate_system_health_report():
    """Generate comprehensive system health report"""
    print("\n📊 Generating system health report...")
    
    try:
        from enhanced_blackboard import get_enhanced_blackboard
        
        ebb = get_enhanced_blackboard()
        await ebb.start()
        
        stats = await ebb.get_comprehensive_stats()
        
        print("\n" + "=" * 50)
        print("📊 PERSISTENT MEMORY SYSTEM HEALTH REPORT")
        print("=" * 50)
        
        # Memory statistics
        if "persistent_memory" in stats:
            mem_stats = stats["persistent_memory"]
            print(f"\n🧠 Memory Statistics:")
            print(f"  Total Memories: {mem_stats.get('total_memories', 0)}")
            print(f"  Storage Efficiency: {mem_stats.get('storage_efficiency', 0):.1%}")
            
            if "by_type" in mem_stats:
                print(f"  By Type:")
                for mem_type, count in mem_stats["by_type"].items():
                    print(f"    {mem_type}: {count}")
        
        # Blackboard statistics
        if "blackboard" in stats:
            bb_stats = stats["blackboard"]
            print(f"\n📋 Blackboard Statistics:")
            if "total_tokens" in bb_stats:
                print(f"  Total Tokens: {bb_stats['total_tokens']}")
                print(f"  Total Operations: {bb_stats['total_operations']}")
        
        # System health
        if "system_health" in stats:
            health = stats["system_health"]
            print(f"\n🔧 System Health:")
            print(f"  Memory Enabled: {health.get('memory_enabled', False)}")
            print(f"  Context Window: {health.get('context_window_hours', 0)} hours")
            print(f"  Max Context Memories: {health.get('max_context_memories', 0)}")
        
        await ebb.stop()
        print("\n✅ System health report complete")
        return True
        
    except Exception as e:
        print(f"❌ Health report error: {e}")
        return False

async def main():
    """Main setup function"""
    print("🚀 Persistent Memory System Setup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Setup failed due to missing dependencies")
        sys.exit(1)
    
    # Check configuration
    if not check_influxdb_config():
        print("\n❌ Setup failed due to configuration issues")
        sys.exit(1)
    
    # Check InfluxDB connection
    if not await check_influxdb_connection():
        print("\n❌ Setup failed due to InfluxDB connection issues")
        sys.exit(1)
    
    # Create memory buckets
    if not await create_memory_buckets():
        print("\n❌ Setup failed during bucket creation")
        sys.exit(1)
    
    # Validate persistent memory
    if not await validate_persistent_memory():
        print("\n❌ Setup failed during persistent memory validation")
        sys.exit(1)
    
    # Validate enhanced blackboard
    if not await validate_enhanced_blackboard():
        print("\n❌ Setup failed during enhanced blackboard validation")
        sys.exit(1)
    
    # Setup agent contexts
    if not await setup_agent_contexts():
        print("\n❌ Setup failed during agent context initialization")
        sys.exit(1)
    
    # Generate health report
    if not await generate_system_health_report():
        print("\n⚠️  Warning: Could not generate health report")
    
    print("\n" + "=" * 50)
    print("🎉 PERSISTENT MEMORY SYSTEM SETUP COMPLETE!")
    print("=" * 50)
    print("\n✅ All components initialized successfully")
    print("✅ Memory buckets created")
    print("✅ Agent contexts configured")
    print("✅ System validation passed")
    print("\n💡 Next steps:")
    print("  1. Run test suite: python test_persistent_memory.py")
    print("  2. Use enhanced blackboard in your agents")
    print("  3. Monitor memory usage and efficiency")
    print("\n📚 Documentation:")
    print("  - Enhanced Blackboard: enhanced_blackboard.py")
    print("  - Persistent Memory: persistent_memory.py")
    print("  - Configuration: config/blackboard_influx.yaml")

if __name__ == "__main__":
    asyncio.run(main())