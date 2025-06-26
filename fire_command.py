#!/usr/bin/env python3
"""
🔥 /fire - Claude Code Slash Command Integration

This script provides /fire slash command functionality for Claude Code,
integrating with the real SPARC trio agents for production-grade development.

Usage from Claude Code:
  /fire "build trading API with authentication"
  /fire --task "implement ML pipeline" --stack pytorch
  /fire --interactive
  /fire --status
"""

import sys
import subprocess
from pathlib import Path

def execute_fire_command(args):
    """Execute the fire command with given arguments."""
    
    project_root = Path(__file__).parent
    fire_script = project_root / "fire"
    
    # Parse arguments
    if not args:
        print("🔥 FIRE Command")
        print("Usage: /fire 'your task description'")
        print("       /fire --interactive")
        print("       /fire --status") 
        print("       /fire --help")
        return
    
    # Convert arguments for the fire script
    if args[0] == "--help":
        print("🔥 FIRE Command Help")
        print("=" * 40)
        print("Usage:")
        print("  /fire 'task description'")
        print("  /fire --interactive")
        print("  /fire --status")
        print("  /fire --help")
        print("\nExamples:")
        print("  /fire 'build a FastAPI REST API with authentication'")
        print("  /fire 'create a machine learning model for stock prediction'")
        print("  /fire 'implement a secure microservices architecture'")
        return
    
    # Prepare command for subprocess
    cmd = ["python3", str(fire_script)]
    
    if args[0] == "--status":
        cmd.append("--status")
    elif args[0] == "--interactive":
        cmd.append("--interactive")
    else:
        # Join all arguments as the task description
        task = " ".join(args)
        cmd.append(task)
    
    print(f"🔥 Executing FIRE command...")
    print("=" * 50)
    
    try:
        # Execute the fire script
        result = subprocess.run(cmd, cwd=project_root, check=False, text=True)
        
        if result.returncode == 0:
            print("\n🎉 FIRE command completed successfully!")
        else:
            print(f"\n⚠️  FIRE command exited with code: {result.returncode}")
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FIRE command failed: {e}")
    except FileNotFoundError:
        print(f"\n❌ Could not find fire script at: {fire_script}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def main():
    """Main entry point for /fire command."""
    
    # Get arguments from command line (excluding script name)
    args = sys.argv[1:]
    
    # Run the fire command
    execute_fire_command(args)

if __name__ == "__main__":
    main()