#!/usr/bin/env python3
"""
FlowState Quick Start Script
Simplified development server startup
"""

import os
import sys
import uvicorn
import logging

def setup_logging():
    """Configure logging for development"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('flowstate.log')
        ]
    )

def main():
    """Start the FlowState development server"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🌊 Starting FlowState Development Server...")
    
    # Development configuration
    config = {
        "app": "main:app",
        "host": "127.0.0.1",
        "port": 8000,
        "reload": True,
        "log_level": "info",
        "access_log": True
    }
    
    try:
        logger.info(f"🚀 Server will start at http://{config['host']}:{config['port']}")
        logger.info("📖 API documentation: http://127.0.0.1:8000/docs")
        logger.info("🔄 Auto-reload enabled for development")
        
        uvicorn.run(**config)
        
    except KeyboardInterrupt:
        logger.info("🛑 FlowState server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
