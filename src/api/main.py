# src/api/main.py
from fastapi import FastAPI, WebSocket, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
import re
import os
import asyncio
import json 
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime

from .websocket_manager import manager
from ..core.pipeline_manager import PipelineManager, StreamSource
from ..core.config import settings, initialize_logger

# Setup logging
logger = initialize_logger()

# Initialize the pipeline manager as a global singleton
pipeline_manager = PipelineManager(
    base_data_dir=settings.PIPELINE.BASE_DATA_DIR,
    max_concurrent_pipelines=settings.PIPELINE.MAX_CONCURRENT_PIPELINES,
    api_key=settings.ANALYSIS.GOOGLE_API_KEY
)

app = FastAPI(title="Multi-Channel Video Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_url(url: str, source_type: str) -> bool:
    """Validate URL format based on source type."""
    if source_type == "youtube":
        patterns = [
            r'^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$',
        ]
        return any(re.match(pattern, url) for pattern in patterns)
    elif source_type == "m3u8":
        # Basic validation for m3u8 URLs
        return url.startswith(('http://', 'https://')) and (url.endswith('.m3u8') or '.m3u8' in url)
    else:
        return False

# Models
class SourceRequest(BaseModel):
    url: str
    source_type: str = "youtube"  # Default to YouTube
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_fields(self) -> 'SourceRequest':
        # Validate URL based on source type
        if not validate_url(self.url, self.source_type):
            raise ValueError(f"Invalid URL format for {self.source_type}")
        
        # Validate source type
        valid_types = settings.PIPELINE.SUPPORTED_SOURCE_TYPES
        if self.source_type not in valid_types:
            raise ValueError(f"Invalid source type. Must be one of: {valid_types}")
            
        return self

class PipelineRequest(BaseModel):
    source_id: str
    chunk_duration: int = settings.PIPELINE.DEFAULT_CHUNK_DURATION
    analysis_prompt: Optional[str] = None
    use_web_search: bool = settings.ANALYSIS.USE_WEB_SEARCH
    export_responses: Optional[bool] = None  
    runtime_duration: int = settings.PIPELINE.DEFAULT_RUNTIME_DURATION 
    
    @model_validator(mode='after')
    def validate_duration(self) -> 'PipelineRequest':
        if (self.chunk_duration < settings.PIPELINE.MIN_CHUNK_DURATION or 
            self.chunk_duration > settings.PIPELINE.MAX_CHUNK_DURATION):
            raise ValueError(
                f"Chunk duration must be between {settings.PIPELINE.MIN_CHUNK_DURATION} "
                f"and {settings.PIPELINE.MAX_CHUNK_DURATION} seconds"
            )
        return self

class PipelineActionRequest(BaseModel):
    pipeline_id: str

# Routes
@app.websocket("/ws/analysis")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)

@app.post("/sources", response_model=Dict[str, Any])
async def create_source(request: SourceRequest):
    """Register a new streaming source."""
    try:
        # Log the full request data
        logger.info(f"Registering new source - URL: {request.url}, type: {request.source_type}")
        logger.debug(f"Full source request data: {request.model_dump_json()}")
        
        source = await pipeline_manager.register_source(
            url=request.url,
            source_type=request.source_type,
            metadata=request.metadata
        )
        
        logger.info(f"Source registered successfully: {source.source_id}")
        logger.debug(f"Source details: {source.model_dump_json()}")
        
        return {
            "status": "success",
            "message": "Source registered successfully",
            "source": source.model_dump()
        }
    except Exception as e:
        logger.error(f"Error registering source: {str(e)}", exc_info=True)
        # Get full traceback for detailed debugging
        tb = traceback.format_exc()
        logger.error(f"Traceback: {tb}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources", response_model=List[Dict[str, Any]])
async def list_sources():
    """List all registered sources."""
    # Get unique sources from all pipelines
    sources = await pipeline_manager.get_all_sources()
    return [source.model_dump() for source in sources]

@app.delete("/sources/{source_id}", response_model=Dict[str, Any])
async def delete_source(source_id: str):
    """Delete a source and all its associated pipelines."""
    # First, add this method to the PipelineManager class
    source = await pipeline_manager.get_source(source_id)
    
    if not source:
        raise HTTPException(
            status_code=404,
            detail=f"Source not found: {source_id}"
        )
    
    # Find all pipelines that use this source
    associated_pipelines = []
    for pid, pipeline in list(pipeline_manager.pipelines.items()):
        if pipeline["source"].source_id == source_id:
            associated_pipelines.append(pid)
            # Stop the pipeline if it's running
            if pipeline["state_machine"].is_active():
                await pipeline_manager.stop_pipeline(pid)
    
    # Remove the source
    async with pipeline_manager._source_lock:
        del pipeline_manager.sources[source_id]
    
    return {
        "status": "success",
        "message": f"Source {source_id} deleted along with {len(associated_pipelines)} associated pipelines",
        "deleted_pipelines": associated_pipelines
    }

@app.post("/pipelines", response_model=Dict[str, Any])
async def create_pipeline(request: PipelineRequest):
    """Create a new pipeline for a source."""
    try:
        # Log the full request data
        logger.info(f"Creating pipeline for source: {request.source_id}")
        logger.debug(f"Full pipeline request data: {request.model_dump_json()}")
        
        # Get the source directly from pipeline_manager
        source = await pipeline_manager.get_source(request.source_id)
        
        if not source:
            logger.error(f"Source not found: {request.source_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Source not found: {request.source_id}"
            )
            
        # Create pipeline with export_responses parameter
        pipeline_id = await pipeline_manager.create_pipeline(
            source=source,  # Pass the actual source object
            chunk_duration=request.chunk_duration,
            analysis_prompt=request.analysis_prompt,
            use_web_search=request.use_web_search,
            export_responses=request.export_responses,
            runtime_duration=request.runtime_duration
        )
        
        logger.info(f"Pipeline created successfully: {pipeline_id}")
        
        return {
            "status": "success",
            "message": "Pipeline created successfully",
            "pipeline_id": pipeline_id
        }
    except ValueError as e:
        # Handle the case where source is not found but caught by pipeline_manager
        logger.error(f"Source error: {str(e)}")
        raise HTTPException(
            status_code=404, 
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating pipeline: {str(e)}", exc_info=True)
        # Get full traceback for detailed debugging
        tb = traceback.format_exc()
        logger.error(f"Traceback: {tb}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipelines/{pipeline_id}/start", response_model=Dict[str, Any])
async def start_pipeline(pipeline_id: str):
    """Start a pipeline."""
    try:
        logger.info(f"Starting pipeline: {pipeline_id}")
        
        success = await pipeline_manager.start_pipeline(pipeline_id)
        
        if not success:
            logger.error(f"Failed to start pipeline: {pipeline_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Cannot start pipeline {pipeline_id}"
            )
        
        logger.info(f"Pipeline started successfully: {pipeline_id}")
        
        return {
            "status": "success",
            "message": "Pipeline started successfully",
            "pipeline_id": pipeline_id
        }
    except Exception as e:
        logger.error(f"Error starting pipeline: {str(e)}", exc_info=True)
        # Get full traceback for detailed debugging
        tb = traceback.format_exc()
        logger.error(f"Traceback: {tb}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pipelines/{pipeline_id}/stop", response_model=Dict[str, Any])
async def stop_pipeline(pipeline_id: str):
    """Stop a pipeline."""
    try:
        logger.info(f"Stoping pipeline: {pipeline_id}")
        
        success = await pipeline_manager.stop_pipeline(pipeline_id)
        
        if not success:
            logger.error(f"Failed to stop pipeline: {pipeline_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Cannot stop pipeline {pipeline_id}"
            )
        
        logger.info(f"Pipeline stoped successfully: {pipeline_id}")
          
        return {
            "status": "success",
            "message": "Pipeline stop signal sent",
            "pipeline_id": pipeline_id
        }
    except Exception as e:
        logger.error(f"Error starting pipeline: {str(e)}", exc_info=True)
        # Get full traceback for detailed debugging
        tb = traceback.format_exc()
        logger.error(f"Traceback: {tb}")    
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipelines/{pipeline_id}", response_model=Dict[str, Any])
async def get_pipeline_status(pipeline_id: str):
    """Get the status of a pipeline."""
    status = await pipeline_manager.get_pipeline_status(pipeline_id)
    
    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline not found: {pipeline_id}"
        )
        
    return status

@app.get("/pipelines", response_model=List[Dict[str, Any]])
async def list_pipelines():
    """List all pipelines."""
    return await pipeline_manager.get_all_pipeline_statuses()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "active_pipelines": sum(1 for p in pipeline_manager.pipelines.values() 
                              if p["state_machine"].is_active()),
        "total_pipelines": len(pipeline_manager.pipelines),
        "version": "2.0.0",
        "analysis_settings": {
            "export_responses": settings.ANALYSIS.EXPORT_RESPONSES,
            "test_mode": os.getenv("GEMINI_TEST_MODE", "").lower() == "true",
            "use_web_search": settings.ANALYSIS.USE_WEB_SEARCH,
            "max_retries": settings.ANALYSIS.MAX_RETRIES
        }
    }

# Shutdown event to clean up resources
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    # Stop all active pipelines
    for pipeline_id in list(pipeline_manager.pipelines.keys()):
        await pipeline_manager.stop_pipeline(pipeline_id)
        
    # Wait for all tasks to complete
    tasks = list(pipeline_manager.tasks.values())
    if tasks:
        done, pending = await asyncio.wait(tasks, timeout=5.0)
        for task in pending:
            task.cancel()
            
@app.get("/settings", response_model=Dict[str, Any])
async def get_settings():
    """Get current application settings (safe fields only)."""
    return {
        "pipeline": {
            "default_chunk_duration": settings.PIPELINE.DEFAULT_CHUNK_DURATION,
            "min_chunk_duration": settings.PIPELINE.MIN_CHUNK_DURATION,
            "max_chunk_duration": settings.PIPELINE.MAX_CHUNK_DURATION,
            "max_concurrent_pipelines": settings.PIPELINE.MAX_CONCURRENT_PIPELINES,
            "supported_source_types": settings.PIPELINE.SUPPORTED_SOURCE_TYPES,
            "default_runtime_duration": settings.PIPELINE.DEFAULT_RUNTIME_DURATION
        },
        "analysis": {
            "use_web_search": settings.ANALYSIS.USE_WEB_SEARCH,
            "export_responses": settings.ANALYSIS.EXPORT_RESPONSES,
            "max_retries": settings.ANALYSIS.MAX_RETRIES,
            "retry_delay": settings.ANALYSIS.RETRY_DELAY,
            "test_mode": os.getenv("GEMINI_TEST_MODE", "").lower() == "true"
        },
        "environment": settings.API.ENVIRONMENT,
        "response_export_path": os.path.join(settings.PIPELINE.BASE_DATA_DIR, "gemini_responses")
    }
    
@app.get("/responses", response_model=List[Dict[str, Any]])
async def list_response_files():
    """List all exported Gemini response files."""
    responses_dir = os.path.join(settings.PIPELINE.BASE_DATA_DIR, "gemini_responses")
    response_files = []
    
    try:
        # Create directory if it doesn't exist
        if not os.path.exists(responses_dir):
            os.makedirs(responses_dir, exist_ok=True)
            return response_files
            
        # List files
        files = os.listdir(responses_dir)
        
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(responses_dir, file)
                stat = os.stat(file_path)
                
                # Parse filename to extract metadata
                parts = file.split('_')
                file_type = "unknown"
                
                if "websearch" in file:
                    file_type = "web_search"
                elif "standard" in file:
                    file_type = "standard"
                
                response_files.append({
                    "filename": file,
                    "path": file_path,
                    "size_bytes": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": file_type
                })
                
        # Sort by created time, newest first
        response_files.sort(key=lambda x: x["created"], reverse=True)
        
        return response_files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing response files: {str(e)}")

@app.get("/responses/{filename}", response_model=Dict[str, Any])
async def get_response_file(filename: str):
    """Get a specific Gemini response file."""
    responses_dir = os.path.join(settings.PIPELINE.BASE_DATA_DIR, "gemini_responses")
    file_path = os.path.join(responses_dir, filename)
    
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Response file not found: {filename}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format in file: {filename}")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error reading response file: {str(e)}")
    

'''
class LegacyAnalysisRequest(BaseModel):
    url: str
    chunk_duration: int = settings.PIPELINE.DEFAULT_CHUNK_DURATION
    export_responses: Optional[bool] = None  # Added field for controlling response export
    runtime_duration: int = settings.PIPELINE.DEFAULT_RUNTIME_DURATION
    
    @model_validator(mode='after')
    def validate_fields(self) -> 'LegacyAnalysisRequest':
        # Validate URL
        if not validate_url(self.url):
            raise ValueError("Invalid YouTube URL format")
            
        # Validate chunk duration
        if (self.chunk_duration < settings.PIPELINE.MIN_CHUNK_DURATION or 
            self.chunk_duration > settings.PIPELINE.MAX_CHUNK_DURATION):
            raise ValueError(
                f"Chunk duration must be between {settings.PIPELINE.MIN_CHUNK_DURATION} "
                f"and {settings.PIPELINE.MAX_CHUNK_DURATION} seconds"
            )
        return self
'''
'''
# Backward compatibility with original API
@app.post("/start-analysis")
async def start_analysis_legacy(request: LegacyAnalysisRequest):
    """Legacy endpoint for starting analysis."""
    try:
        # Register source
        source = await pipeline_manager.register_source(
            url=request.url,
            source_type="youtube",
            metadata={"chunk_duration": request.chunk_duration}
        )
        
        # Create pipeline with export_responses parameter
        pipeline_id = await pipeline_manager.create_pipeline(
            source=source,
            chunk_duration=request.chunk_duration,
            analysis_prompt=settings.ANALYSIS.DEFAULT_ANALYSIS_PROMPT,
            use_web_search=settings.ANALYSIS.USE_WEB_SEARCH,
            export_responses=request.export_responses,
            runtime_duration=request.runtime_duration
        )
        
        # Start pipeline
        await pipeline_manager.start_pipeline(pipeline_id)
        
        return {
            "status": "success",
            "message": "Analysis started",
            "pipeline_id": pipeline_id,
            "url": request.url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop-analysis")
async def stop_analysis_legacy(request: LegacyAnalysisRequest):
    """Legacy endpoint for stopping analysis."""
    try:
        # Find pipeline by URL
        pipeline_id = None
        for pid, pipeline in pipeline_manager.pipelines.items():
            if pipeline["source"].url == request.url:
                pipeline_id = pid
                break
                
        if not pipeline_id:
            raise HTTPException(
                status_code=404,
                detail=f"No active analysis found for URL: {request.url}"
            )
            
        # Stop pipeline
        await pipeline_manager.stop_pipeline(pipeline_id)
        
        return {
            "status": "success",
            "message": "Analysis stopped",
            "pipeline_id": pipeline_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
