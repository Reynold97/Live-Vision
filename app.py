# app.py
import asyncio
import logging
import os
import uvicorn
from contextlib import asynccontextmanager

from src.core.config import settings, initialize_logger
from src.api.main import app as api_app

# Initialize logger
logger = initialize_logger()

def validate_dependencies():
    """Validate that required dependencies are available."""
    try:
        import yt_dlp
        import google.genai
        import watchdog
    except ImportError as e:
        logger.critical(f"Missing dependency: {e}")
        raise SystemExit(f"Missing dependency: {e}")
        
    # Check if ffmpeg is installed
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.critical("ffmpeg is not installed or not in PATH")
            raise SystemExit("ffmpeg is not installed or not in PATH")
    except FileNotFoundError:
        logger.critical("ffmpeg is not installed or not in PATH")
        raise SystemExit("ffmpeg is not installed or not in PATH")
        
    logger.info("All dependencies validated")

def create_directories():
    """Create required directories."""
    directories = [
        settings.PIPELINE.BASE_DATA_DIR,
        settings.LOGGING.LOG_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created (if not exists): {directory}")

def check_api_keys():
    """Check if required API keys are configured."""
    if not settings.ANALYSIS.GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY is not set. Analysis functionality will not work.")
    else:
        logger.info("API keys validated")

def create_application():
    """Create and configure the FastAPI application."""
    # Use the existing app from api/main.py
    return api_app

@asynccontextmanager
async def lifespan(app):
    """
    Lifespan manager for the API application.
    Handles startup and shutdown tasks.
    """
    # Startup
    logger.info("Application starting up...")
    yield
    # Shutdown
    logger.info("Application shutting down...")

def main():
    """Main application entry point."""
    # Banner
    print("""
    ***************************************
    *  Multi-Channel Stream Analysis API  *
    ***************************************
    """)
    
    # Validate environment
    validate_dependencies()
    create_directories()
    check_api_keys()
    
    # Configure the FastAPI app
    app = create_application()
    
    # Add lifespan manager if running a recent version of FastAPI
    try:
        app.router.lifespan_context = lifespan
    except AttributeError:
        # For older FastAPI versions, we'll use the older lifespan approach
        from fastapi import FastAPI
        if hasattr(FastAPI, 'lifespan'):
            app.lifespan = lifespan
    
    # Log settings
    logger.info(f"Environment: {settings.API.ENVIRONMENT}")
    logger.info(f"DEBUG mode: {settings.API.DEBUG}")
    logger.info(f"Max concurrent pipelines: {settings.PIPELINE.MAX_CONCURRENT_PIPELINES}")
    
    # Start the server
    logger.info(f"Starting server on {settings.API.HOST}:{settings.API.PORT}")
    
    uvicorn.run(
        app,
        host=settings.API.HOST,
        port=settings.API.PORT,
        log_level=settings.LOGGING.LOG_LEVEL.lower(),
        reload=settings.API.ENVIRONMENT == "development"
    )

if __name__ == "__main__":
    main()