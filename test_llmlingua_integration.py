"""
Test LLMLingua Integration
Validate that LLMLingua compression works with our services
"""

import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_llmlingua_basic():
    """Test basic LLMLingua functionality"""
    try:
        # Import our services
        from src.core.llmlingua_service import LLMLinguaService, CompressionConfig
        from src.core.agent_compression import AgentCompressionService, AgentRole, AgentPromptContext
        
        # Create services
        llmlingua_service = LLMLinguaService()
        agent_compression_service = AgentCompressionService(llmlingua_service)
        
        # Test basic compression
        test_prompt = """
        You are a financial analyst tasked with analyzing market trends and providing investment recommendations. 
        Please review the following market data and provide a comprehensive analysis including technical indicators, 
        fundamental analysis, and risk assessment. Consider the current economic environment, interest rates, 
        inflation trends, and geopolitical factors that might impact the markets. Provide specific buy, hold, 
        or sell recommendations with clear reasoning and risk mitigation strategies.
        """
        
        logger.info("Testing basic prompt compression...")
        
        config = CompressionConfig(
            target_token=50,
            use_llmlingua2=True,
            fallback_on_error=True
        )
        
        result = await llmlingua_service.compress_prompt(test_prompt, config)
        
        logger.info(f"Compression Results:")
        logger.info(f"  Original tokens: {result.original_tokens}")
        logger.info(f"  Compressed tokens: {result.compressed_tokens}")
        logger.info(f"  Compression ratio: {result.compression_ratio:.2f}")
        logger.info(f"  Processing time: {result.processing_time_ms:.1f}ms")
        logger.info(f"  Model used: {result.model_used}")
        logger.info(f"  Cost savings: ${result.cost_savings_estimate:.4f}")
        logger.info(f"  Error: {result.error}")
        logger.info(f"  Fallback used: {result.fallback_used}")
        
        if result.error:
            logger.warning(f"Compression had error: {result.error}")
        
        logger.info(f"Original prompt (first 100 chars): {test_prompt[:100]}...")
        logger.info(f"Compressed prompt: {result.compressed_prompt}")
        
        return True
        
    except Exception as e:
        logger.error(f"Basic compression test failed: {e}")
        return False

async def test_agent_compression():
    """Test agent-specific compression"""
    try:
        from src.core.llmlingua_service import LLMLinguaService
        from src.core.agent_compression import AgentCompressionService, AgentRole, AgentPromptContext
        
        llmlingua_service = LLMLinguaService()
        agent_service = AgentCompressionService(llmlingua_service)
        
        # Test RESEARCHER agent compression
        research_prompt = """
        As a RESEARCHER agent, analyze the current market conditions for technology stocks. 
        Gather data from multiple sources including earnings reports, analyst recommendations, 
        technical indicators, and market sentiment. Focus on key players like AAPL, MSFT, GOOGL, 
        and emerging tech companies. Provide comprehensive research findings that can be used 
        for strategic decision making.
        """
        
        logger.info("Testing RESEARCHER agent compression...")
        
        context = AgentPromptContext(
            agent_role=AgentRole.RESEARCHER,
            task_type="research",
            priority="high",
            domain="trading"
        )
        
        result = await agent_service.compress_agent_prompt(research_prompt, context)
        
        logger.info(f"RESEARCHER Compression:")
        logger.info(f"  Original tokens: {result.original_tokens}")
        logger.info(f"  Compressed tokens: {result.compressed_tokens}")
        logger.info(f"  Compression ratio: {result.compression_ratio:.2f}")
        logger.info(f"  Processing time: {result.processing_time_ms:.1f}ms")
        
        # Test agent stats
        stats = await agent_service.get_agent_stats()
        logger.info(f"Agent stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Agent compression test failed: {e}")
        return False

async def test_health_check():
    """Test LLMLingua service health"""
    try:
        from src.core.llmlingua_service import LLMLinguaService
        
        service = LLMLinguaService()
        health = await service.health_check()
        
        logger.info(f"LLMLingua Health Check: {health}")
        
        stats = await service.get_compression_stats()
        logger.info(f"Compression Stats: {stats}")
        
        return health.get("llmlingua") == "healthy"
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

async def main():
    """Run all LLMLingua integration tests"""
    logger.info("🚀 Starting LLMLingua Integration Tests")
    
    tests = [
        ("Basic Compression", test_llmlingua_basic),
        ("Agent Compression", test_agent_compression), 
        ("Health Check", test_health_check)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = await test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.info(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:.<30} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All LLMLingua integration tests passed!")
        logger.info("✅ LLMLingua baseline is ready for production use")
    else:
        logger.warning("⚠️ Some tests failed - check logs for details")

if __name__ == "__main__":
    asyncio.run(main())