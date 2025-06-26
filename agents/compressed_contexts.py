
# ADDED: Ultra-Compressed Agent Contexts
COMPRESSED_AGENT_CONTEXTS = {
    "researcher": {
        "core": "R:MI,TA,SI,PB|Q:comprehensive|F:evidence-based",
        "capabilities": ["intelligence", "analysis", "security"],
        "token_budget": 1000,
        "focus": "data-driven insights"
    },
    
    "mastermind": {
        "core": "M:SA,AA,QS,RA|T:strategic|P:architecture", 
        "capabilities": ["strategy", "design", "quality"],
        "token_budget": 1500,
        "focus": "systematic approach"
    },
    
    "executor": {
        "core": "E:TDD,QV,DP,95%|M:production|S:secure",
        "capabilities": ["implementation", "testing", "deployment"],
        "token_budget": 1200,
        "focus": "quality delivery"
    }
}

class CompressedContextLoader:
    """Load minimal agent contexts based on specific needs"""
    
    def __init__(self):
        self.context_cache = {}
    
    def get_context(self, agent: str, operation: str = None) -> Dict:
        """Get minimal context for agent/operation combination"""
        
        cache_key = f"{agent}:{operation}" if operation else agent
        
        if cache_key in self.context_cache:
            return self.context_cache[cache_key]
        
        # Get base compressed context
        base_context = COMPRESSED_AGENT_CONTEXTS.get(agent.lower(), {})
        
        # Add operation-specific context if needed
        if operation:
            operation_context = self._get_operation_context(operation)
            context = {**base_context, **operation_context}
        else:
            context = base_context
        
        # Cache for reuse
        self.context_cache[cache_key] = context
        return context
    
    def _get_operation_context(self, operation: str) -> Dict:
        """Get minimal context for specific operations"""
        
        operation_contexts = {
            "technical_analysis": {
                "indicators": ["RSI", "MACD"],
                "timeframes": ["1h", "4h", "1d"],
                "focus": "trends"
            },
            
            "market_intelligence": {
                "sources": ["news", "social", "volume"],
                "refresh": 300,
                "focus": "sentiment"
            },
            
            "strategic_analysis": {
                "frameworks": ["SPARC", "SWOT"],
                "depth": "comprehensive",
                "focus": "decisions"
            },
            
            "implementation": {
                "methodology": "TDD",
                "coverage": 95,
                "focus": "quality"
            }
        }
        
        return operation_contexts.get(operation, {})
    
    def get_context_string(self, agent: str, operation: str = None) -> str:
        """Get context as compressed string for prompts"""
        context = self.get_context(agent, operation)
        
        # Convert to ultra-compact string
        if "core" in context:
            return context["core"]
        
        # Fallback format
        capabilities = ",".join(context.get("capabilities", [])[:3])
        focus = context.get("focus", "general")
        return f"{agent[0].upper()}:{capabilities}|F:{focus}"

# Global instance
compressed_context_loader = CompressedContextLoader()

def get_compressed_context(agent: str, operation: str = None) -> str:
    """Quick function to get compressed context string"""
    return compressed_context_loader.get_context_string(agent, operation)
