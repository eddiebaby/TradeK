"""
OpenAI Quota Management Utility

Command-line tool for monitoring and managing OpenAI embedding quota usage.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingestion.hybrid_embedding_router import OpenAIQuotaManager


async def show_quota_status():
    """Display current quota status"""
    quota_manager = OpenAIQuotaManager()
    
    print("🔍 OpenAI Embedding Quota Status")
    print("=" * 40)
    
    usage_pct = quota_manager.get_usage_percentage()
    remaining_gb = quota_manager.usage.remaining_quota / (1024 * 1024 * 1024)
    used_gb = quota_manager.usage.total_size_bytes / (1024 * 1024 * 1024)
    
    print(f"📊 Usage: {usage_pct:.1f}% ({used_gb:.3f} GB / 1.0 GB)")
    print(f"💾 Remaining: {remaining_gb:.3f} GB")
    print(f"📄 Documents processed: {quota_manager.usage.documents_processed:,}")
    print(f"🔤 Total tokens: {quota_manager.usage.total_tokens:,}")
    print(f"📅 Last usage: {quota_manager.usage.last_usage}")
    
    if usage_pct > 80:
        print("\n⚠️  WARNING: Quota usage is high!")
    elif usage_pct > 95:
        print("\n🚨 CRITICAL: Quota nearly exhausted!")
    else:
        print("\n✅ Quota usage is healthy")


async def reset_quota():
    """Reset quota tracking (use with caution)"""
    print("⚠️  Are you sure you want to reset quota tracking? (y/N): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        quota_file = Path("./data/openai_quota.json")
        if quota_file.exists():
            quota_file.unlink()
            print("✅ Quota tracking reset")
        else:
            print("ℹ️  No quota file found")
    else:
        print("❌ Reset cancelled")


async def estimate_capacity(file_path: str):
    """Estimate how much content can be processed with remaining quota"""
    quota_manager = OpenAIQuotaManager()
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    file_size = len(content.encode('utf-8'))
    remaining_quota = quota_manager.usage.remaining_quota
    
    if file_size <= remaining_quota:
        files_possible = remaining_quota // file_size
        print(f"✅ Can process {files_possible} files like '{file_path}'")
    else:
        print(f"❌ File too large for remaining quota")
        print(f"   File size: {file_size:,} bytes")
        print(f"   Remaining: {remaining_quota:,} bytes")


async def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("OpenAI Quota Manager")
        print("Usage:")
        print("  python quota_manager.py status       - Show quota status")
        print("  python quota_manager.py reset        - Reset quota tracking")
        print("  python quota_manager.py estimate <file> - Estimate capacity")
        return
    
    command = sys.argv[1]
    
    if command == "status":
        await show_quota_status()
    elif command == "reset":
        await reset_quota()
    elif command == "estimate" and len(sys.argv) > 2:
        await estimate_capacity(sys.argv[2])
    else:
        print(f"❌ Unknown command: {command}")


if __name__ == "__main__":
    asyncio.run(main())