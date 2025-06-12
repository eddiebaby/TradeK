#!/usr/bin/env python3
"""
TradeKnowledge - Main Application Entry Point
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.config import load_config, Config
from utils.logging import setup_logging

# Load configuration
config: Config = load_config()

# Setup logging
logger = setup_logging(config.app.log_level)

# Create FastAPI app
app = FastAPI(
    title=config.app.name,
    version=config.app.version,
    description="Book Knowledge MCP Server for Algorithmic Trading"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info(f"Starting {config.app.name} v{config.app.version}")
    # TODO: Initialize MCP Server when created
    logger.info("Server initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down server")
    # TODO: Cleanup MCP server when created

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": config.app.name,
        "version": config.app.version,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": config.app.name,
        "version": config.app.version
    }

def main():
    """Main entry point"""
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.app.debug,
        workers=1 if config.app.debug else config.server.workers,
        log_level=config.app.log_level.lower()
    )

if __name__ == "__main__":
    main()