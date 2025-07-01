#!/usr/bin/env python3
"""
SuperClaude Integration Service
Syncs conversations and templates between SuperClaude Chrome extension and TradeKnowledge SPARC agents
"""

import asyncio
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import hashlib
import shutil
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.core.llmlingua_service import LLMLinguaService, CompressionConfig
except ImportError:
    print("⚠️  LLMLingua service not available, running without compression")
    LLMLinguaService = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SuperClaudeSync:
    """SuperClaude integration service for TradeKnowledge"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize SuperClaude sync service"""
        self.config = config or self._load_config()
        
        # Directory paths
        self.export_dir = Path(self.config.get("export_dir", "~/Downloads/superclaude_exports")).expanduser()
        self.conversations_dir = Path(self.config.get("conversations_dir", "~/TradeKnowledge/agents/conversations")).expanduser()
        self.templates_dir = Path(self.config.get("templates_dir", "~/TradeKnowledge/agents/templates")).expanduser()
        self.processed_dir = self.export_dir / "processed"
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Agent classification keywords
        self.agent_keywords = {
            "RESEARCHER": ["research", "analyze", "investigate", "data", "market", "study", "explore", "examine"],
            "MASTERMIND": ["strategy", "architecture", "design", "plan", "structure", "organize", "coordinate", "manage"],
            "EXECUTOR": ["implement", "code", "test", "deploy", "build", "execute", "develop", "create"]
        }
        
        # Service state
        self.processed_files = set()
        self.stats = {
            "total_imported": 0,
            "classified_conversations": {"RESEARCHER": 0, "MASTERMIND": 0, "EXECUTOR": 0, "UNCLASSIFIED": 0},
            "templates_exported": 0,
            "errors": 0,
            "last_sync": None
        }
        
        # Initialize LLMLingua if available
        self.llmlingua_service = LLMLinguaService() if LLMLinguaService else None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        config_file = Path("~/.claude/superclaude_config.json").expanduser()
        
        default_config = {
            "export_dir": "~/Downloads/superclaude_exports",
            "conversations_dir": "~/TradeKnowledge/agents/conversations",
            "templates_dir": "~/TradeKnowledge/agents/templates",
            "auto_sync_interval": 300,  # 5 minutes
            "enable_compression": True,
            "classification_threshold": 0.6,
            "max_conversation_age_days": 30,
            "backup_processed_files": True
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config file, using defaults: {e}")
        
        return default_config
    
    def _ensure_directories(self):
        """Create necessary directories"""
        for directory in [self.export_dir, self.conversations_dir, self.templates_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file to track processing"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    async def classify_conversation(self, conversation_text: str) -> str:
        """Classify conversation by SPARC agent type using keyword analysis and LLMLingua"""
        
        # Compress text if LLMLingua is available for better analysis
        if self.llmlingua_service:
            try:
                await self.llmlingua_service.initialize()
                result = await self.llmlingua_service.compress_prompt(
                    conversation_text,
                    CompressionConfig(target_token=200, enable_caching=False)
                )
                analysis_text = result.compressed_prompt
            except Exception as e:
                logger.warning(f"LLMLingua compression failed, using original text: {e}")
                analysis_text = conversation_text
        else:
            analysis_text = conversation_text.lower()
        
        # Score each agent type based on keyword frequency
        scores = {}
        total_words = len(analysis_text.split())
        
        for agent_type, keywords in self.agent_keywords.items():
            keyword_count = sum(analysis_text.lower().count(keyword) for keyword in keywords)
            scores[agent_type] = keyword_count / max(total_words, 1)
        
        # Find the best match
        best_agent = max(scores, key=scores.get)
        best_score = scores[best_agent]
        
        # Apply threshold
        threshold = self.config.get("classification_threshold", 0.01)
        if best_score >= threshold:
            logger.info(f"Classified as {best_agent} (score: {best_score:.3f})")
            return best_agent
        else:
            logger.info(f"Could not classify conversation (best score: {best_score:.3f})")
            return "UNCLASSIFIED"
    
    async def process_conversation_file(self, file_path: Path) -> bool:
        """Process a single SuperClaude conversation export file"""
        try:
            logger.info(f"Processing conversation file: {file_path.name}")
            
            # Check if already processed
            file_hash = self._get_file_hash(file_path)
            if file_hash in self.processed_files:
                logger.debug(f"File {file_path.name} already processed, skipping")
                return True
            
            # Load conversation data
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            # Extract conversation text
            conversation_text = self._extract_conversation_text(conversation_data)
            if not conversation_text:
                logger.warning(f"No conversation text found in {file_path.name}")
                return False
            
            # Classify conversation
            agent_type = await self.classify_conversation(conversation_text)
            
            # Prepare output data
            processed_conversation = {
                "source_file": str(file_path),
                "import_timestamp": datetime.now().isoformat(),
                "agent_classification": agent_type,
                "original_data": conversation_data,
                "conversation_text": conversation_text,
                "file_hash": file_hash
            }
            
            # Save to appropriate agent directory
            output_dir = self.conversations_dir / agent_type.lower()
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"superclaude_{file_path.stem}_{int(time.time())}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_conversation, f, indent=2, ensure_ascii=False)
            
            # Move original to processed directory if configured
            if self.config.get("backup_processed_files", True):
                processed_file = self.processed_dir / file_path.name
                shutil.move(str(file_path), str(processed_file))
                logger.info(f"Moved {file_path.name} to processed directory")
            
            # Update statistics
            self.stats["total_imported"] += 1
            self.stats["classified_conversations"][agent_type] += 1
            self.processed_files.add(file_hash)
            
            logger.info(f"Successfully imported conversation to {agent_type} agent")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            self.stats["errors"] += 1
            return False
    
    def _extract_conversation_text(self, conversation_data: Dict[str, Any]) -> str:
        """Extract conversation text from SuperClaude export format"""
        try:
            # Handle different SuperClaude export formats
            if "messages" in conversation_data:
                # Standard message format
                messages = []
                for message in conversation_data["messages"]:
                    role = message.get("role", "unknown")
                    content = message.get("content", "")
                    messages.append(f"{role}: {content}")
                return "\n\n".join(messages)
            
            elif "conversation" in conversation_data:
                # Alternative format
                return conversation_data["conversation"]
            
            elif "content" in conversation_data:
                # Simple content format
                return conversation_data["content"]
            
            else:
                # Try to find any text content
                text_content = []
                for key, value in conversation_data.items():
                    if isinstance(value, str) and len(value) > 50:
                        text_content.append(value)
                return "\n\n".join(text_content)
                
        except Exception as e:
            logger.error(f"Failed to extract conversation text: {e}")
            return ""
    
    async def sync_conversations(self) -> Dict[str, int]:
        """Sync all new conversations from SuperClaude exports"""
        logger.info("Starting conversation sync...")
        
        # Find all JSON files in export directory
        export_files = list(self.export_dir.glob("*.json"))
        new_files = [f for f in export_files if self._get_file_hash(f) not in self.processed_files]
        
        if not new_files:
            logger.info("No new conversation files found")
            return {"processed": 0, "errors": 0}
        
        logger.info(f"Found {len(new_files)} new conversation files")
        
        # Process each file
        processed_count = 0
        error_count = 0
        
        for file_path in new_files:
            try:
                success = await self.process_conversation_file(file_path)
                if success:
                    processed_count += 1
                else:
                    error_count += 1
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                error_count += 1
        
        self.stats["last_sync"] = datetime.now().isoformat()
        
        logger.info(f"Sync complete: {processed_count} processed, {error_count} errors")
        return {"processed": processed_count, "errors": error_count}
    
    async def export_sparc_templates(self) -> int:
        """Export SPARC agent prompts as SuperClaude templates"""
        logger.info("Exporting SPARC templates...")
        
        templates = {
            "RESEARCHER Market Analysis": {
                "description": "RESEARCHER agent template for market analysis",
                "template": "🔍 RESEARCHER Agent - Market Analysis\n\nAnalyze the following market conditions:\n{market_data}\n\nFocus areas:\n- Trend analysis\n- Volume patterns\n- Support/resistance levels\n- Risk assessment\n\nProvide comprehensive research findings with data-driven insights.",
                "variables": ["market_data"],
                "category": "TradeKnowledge-Research"
            },
            
            "MASTERMIND Strategy Design": {
                "description": "MASTERMIND agent template for strategic planning",
                "template": "🧠 MASTERMIND Agent - Strategy Design\n\nDevelop strategic architecture for:\n{objective}\n\nConsiderations:\n- System architecture\n- Risk management\n- Resource allocation\n- Quality gates\n- Success metrics\n\nProvide comprehensive strategic framework with implementation roadmap.",
                "variables": ["objective"],
                "category": "TradeKnowledge-Strategy"
            },
            
            "EXECUTOR Implementation": {
                "description": "EXECUTOR agent template for TDD implementation",
                "template": "⚡ EXECUTOR Agent - Implementation\n\nImplement the following using TDD approach:\n{requirements}\n\nDeliverables:\n- Test-driven development (Red-Green-Refactor)\n- Comprehensive test suite\n- Production-ready code\n- Quality gates validation\n- Deployment configuration\n\nEnsure 95%+ test coverage and security compliance.",
                "variables": ["requirements"],
                "category": "TradeKnowledge-Implementation"
            }
        }
        
        # Save templates in SuperClaude format
        templates_exported = 0
        for name, template_data in templates.items():
            template_file = self.templates_dir / f"{name.lower().replace(' ', '_')}_superclaude.json"
            
            superclaude_template = {
                "name": name,
                "description": template_data["description"],
                "template": template_data["template"],
                "variables": template_data["variables"],
                "category": template_data["category"],
                "tags": ["sparc", "tradeknowledge", name.split()[0].lower()],
                "created_by": "TradeKnowledge SPARC System",
                "created_at": datetime.now().isoformat()
            }
            
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(superclaude_template, f, indent=2, ensure_ascii=False)
            
            templates_exported += 1
        
        self.stats["templates_exported"] = templates_exported
        logger.info(f"Exported {templates_exported} SPARC templates")
        return templates_exported
    
    async def health_check(self) -> Dict[str, Any]:
        """Check SuperClaude integration health"""
        health = {
            "status": "healthy",
            "export_directory": {
                "exists": self.export_dir.exists(),
                "writable": self.export_dir.is_dir() and os.access(self.export_dir, os.W_OK),
                "path": str(self.export_dir)
            },
            "conversations_directory": {
                "exists": self.conversations_dir.exists(),
                "writable": self.conversations_dir.is_dir() and os.access(self.conversations_dir, os.W_OK),
                "path": str(self.conversations_dir)
            },
            "llmlingua_service": {
                "available": self.llmlingua_service is not None,
                "initialized": False
            },
            "statistics": self.stats
        }
        
        # Test LLMLingua if available
        if self.llmlingua_service:
            try:
                await self.llmlingua_service.initialize()
                health["llmlingua_service"]["initialized"] = True
            except Exception as e:
                health["llmlingua_service"]["error"] = str(e)
        
        # Check for any critical issues
        critical_issues = []
        if not health["export_directory"]["exists"]:
            critical_issues.append("Export directory does not exist")
        if not health["conversations_directory"]["writable"]:
            critical_issues.append("Cannot write to conversations directory")
        
        if critical_issues:
            health["status"] = "unhealthy"
            health["issues"] = critical_issues
        
        return health
    
    async def auto_sync_daemon(self):
        """Run continuous sync daemon"""
        logger.info("Starting SuperClaude auto-sync daemon...")
        interval = self.config.get("auto_sync_interval", 300)
        
        while True:
            try:
                # Sync conversations
                result = await self.sync_conversations()
                if result["processed"] > 0:
                    logger.info(f"Auto-sync: imported {result['processed']} conversations")
                
                # Export templates periodically (every hour)
                if not self.stats["last_sync"] or \
                   (datetime.now() - datetime.fromisoformat(self.stats["last_sync"])) > timedelta(hours=1):
                    await self.export_sparc_templates()
                
            except Exception as e:
                logger.error(f"Auto-sync error: {e}")
                self.stats["errors"] += 1
            
            # Wait for next sync
            await asyncio.sleep(interval)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="SuperClaude Integration Service")
    parser.add_argument("--sync-once", action="store_true", help="Run sync once and exit")
    parser.add_argument("--daemon", action="store_true", help="Run as continuous sync daemon")
    parser.add_argument("--export-templates", action="store_true", help="Export SPARC templates")
    parser.add_argument("--health", action="store_true", help="Check integration health")
    parser.add_argument("--test", action="store_true", help="Run integration test")
    parser.add_argument("--setup", action="store_true", help="Setup directories and configuration")
    parser.add_argument("--import", dest="import_file", help="Import specific conversation file")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Load custom config if specified
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize service
    sync_service = SuperClaudeSync(config)
    
    async def run_async():
        if args.setup:
            logger.info("Setting up SuperClaude integration...")
            sync_service._ensure_directories()
            await sync_service.export_sparc_templates()
            logger.info("Setup complete!")
            
        elif args.health:
            health = await sync_service.health_check()
            print(json.dumps(health, indent=2))
            
        elif args.test:
            logger.info("Running SuperClaude integration test...")
            health = await sync_service.health_check()
            if health["status"] == "healthy":
                logger.info("✅ SuperClaude integration test passed")
            else:
                logger.error("❌ SuperClaude integration test failed")
                sys.exit(1)
                
        elif args.export_templates:
            count = await sync_service.export_sparc_templates()
            logger.info(f"Exported {count} templates")
            
        elif args.import_file:
            import_path = Path(args.import_file)
            if import_path.exists():
                success = await sync_service.process_conversation_file(import_path)
                if success:
                    logger.info("Import successful")
                else:
                    logger.error("Import failed")
                    sys.exit(1)
            else:
                logger.error(f"File not found: {import_path}")
                sys.exit(1)
                
        elif args.daemon:
            await sync_service.auto_sync_daemon()
            
        elif args.sync_once:
            result = await sync_service.sync_conversations()
            logger.info(f"Sync complete: {result}")
            
        else:
            parser.print_help()
    
    # Run async function
    asyncio.run(run_async())


if __name__ == "__main__":
    import os
    main()