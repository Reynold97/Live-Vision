from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .websocket_manager import manager
from ..pipeline.video_pipeline import VideoPipeline
from ..recorder.stream_recorder import YouTubeChunker
from ..analyzer.gemini_analyzer import GeminiVideoAnalyzer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Live Video Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active pipelines
active_pipelines = {}

class AnalysisRequest(BaseModel):
    url: str
    chunk_duration: int

class StopRequest(BaseModel):
    url: str

@app.websocket("/ws/analysis")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)

@app.post("/start-analysis")
async def start_analysis(request: AnalysisRequest):
    try:
        # Validate chunk duration
        if request.chunk_duration < 10 or request.chunk_duration > 300:
            raise HTTPException(
                status_code=400, 
                detail="Chunk duration must be between 10 and 300 seconds"
            )
        
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500, 
                detail="GOOGLE_API_KEY not found in environment variables"
            )
            
        # Initialize components
        chunker = YouTubeChunker()
        analyzer = GeminiVideoAnalyzer(api_key=api_key)
        
        # Create pipeline
        pipeline = VideoPipeline(
            chunker=chunker,
            analyzer=analyzer
        )
        
        # Store the pipeline
        active_pipelines[request.url] = pipeline
        
        # Start pipeline as background task
        import asyncio
        asyncio.create_task(pipeline.start_pipeline(
            url=request.url,
            chunk_duration=request.chunk_duration,
            analysis_prompt="""
                Analyze this video segment and focus on identifying commercial elements and opportunities. Please:

                1. First describe the main content and context of this segment:
                - What type of event or content is this?
                - Who or what is being shown?
                - What is the setting or location?

                2. Identify specific commercial elements:
                - Teams, athletes, or performers involved
                - Venues or locations
                - Equipment or gear being used
                - Brands or logos visible
                - Any merchandise already being displayed

                3. Search the web and provide:
                - Official merchandise stores for any teams/performers identified
                - Related products with direct purchase links
                - Licensed merchandise availability
                - Similar products or alternatives

                Format your response in markdown with clear sections and include specific URLs where available. 
                Keep your analysis focused on legitimate, official merchandise and commercial opportunities.
                If you cannot identify specific commercial elements, provide context about the general category of products related to the content.
                Don't make any introduction or conclusions, just straight to the points.
                """,
            use_web_search=True
        ))
        
        return {
            "status": "success", 
            "message": "Analysis started",
            "url": request.url,
            "chunk_duration": request.chunk_duration
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop-analysis")
async def stop_analysis(request: StopRequest):
    try:
        pipeline = active_pipelines.get(request.url)
        if pipeline:
            # Stop the pipeline
            pipeline.is_running = False  # Signal the pipeline to stop
            await pipeline._cleanup()    # Run cleanup
            del active_pipelines[request.url]  # Remove from active pipelines
            return {"status": "success", "message": "Analysis stopped"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No active analysis found for URL: {request.url}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Shutdown event to cleanup resources
@app.on_event("shutdown")
async def shutdown_event():
    # Stop all active pipelines
    for pipeline in active_pipelines.values():
        await pipeline._cleanup()
    active_pipelines.clear()