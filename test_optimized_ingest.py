#!/usr/bin/env python3
"""
Test script for optimized PDF ingestion with memory monitoring
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestion.enhanced_book_processor import EnhancedBookProcessor
from ingestion.resource_monitor import ResourceMonitor, ResourceLimits

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_optimized_ingestion():
    """Test optimized PDF ingestion"""
    
    pdf_path = "data/books/Yves Hilpisch - Python for Algorithmic Trading_ From Idea to Cloud Deployment-O'Reilly Media (2020).pdf"
    
    if not Path(pdf_path).exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return False
    
    # Configure resource limits for WSL2
    limits = ResourceLimits(
        max_memory_percent=75.0,  # More conservative for WSL2
        max_memory_mb=1500,       # Limit to 1.5GB
        warning_threshold=60.0,   # Warn earlier
        check_interval=3.0        # Check more frequently
    )
    
    # Initialize processor
    processor = EnhancedBookProcessor()
    
    # Set up custom resource monitor with stricter limits
    processor.resource_monitor = ResourceMonitor(limits)
    
    try:
        logger.info("Initializing enhanced book processor...")
        await processor.initialize()
        
        logger.info(f"Starting optimized ingestion of: {pdf_path}")
        
        # Add resource monitoring callback for detailed logging
        async def detailed_callback(check):
            usage = check['usage']
            logger.info(f"Memory: {usage['system_used_percent']:.1f}% system, "
                       f"{usage['process_memory_mb']:.1f}MB process")
            
            if check['memory_warning']:
                logger.warning(f"Memory warning: {check['recommendations']}")
            
            if check['memory_critical']:
                logger.error(f"Memory critical: {check['recommendations']}")
        
        processor.resource_monitor.add_callback(detailed_callback)
        
        # Process the file
        result = await processor.add_book(
            pdf_path,
            metadata={
                'categories': ['algorithmic-trading', 'python'],
                'description': 'Comprehensive guide to Python for algorithmic trading'
            },
            force_reprocess=True  # Force reprocess for testing
        )
        
        if result['success']:
            logger.info(f"✅ Successfully processed: {result['title']}")
            logger.info(f"📊 Chunks created: {result['chunks_created']}")
            logger.info(f"⏱️  Processing time: {result['processing_time']:.1f}s")
            logger.info(f"🔍 OCR used: {result['ocr_used']}")
            
            # Show content analysis
            analysis = result['content_analysis']
            logger.info(f"📈 Content analysis: {analysis['code_blocks']} code blocks, "
                       f"{analysis['formulas']} formulas, {analysis['tables']} tables")
            
            return True
            
        else:
            logger.error(f"❌ Processing failed: {result['error']}")
            if 'details' in result:
                logger.error(f"Details: {result['details']}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Exception during processing: {e}", exc_info=True)
        return False
    
    finally:
        try:
            await processor.cleanup()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main function"""
    logger.info("🚀 Starting optimized PDF ingestion test")
    
    success = await test_optimized_ingestion()
    
    if success:
        logger.info("✅ Test completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())