# src/core/pipeline_manager.py
import asyncio
import logging
import os
import uuid
import time 
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from pydantic import BaseModel, Field

from .state_machine import PipelineStateMachine, PipelineState, StateChangeEvent
from ..analyzer.gemini_analyzer import GeminiVideoAnalyzer
from ..recorder.stream_recorder import YouTubeChunker
from ..core.config import settings

class StreamSource(BaseModel):
    """Represents a streaming source (YouTube, TV, etc)."""
    source_id: str
    url: str
    source_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class PipelineStats(BaseModel):
    """Statistics for a pipeline."""
    chunks_processed: int = 0
    errors: int = 0
    total_processing_time: float = 0
    avg_processing_time: float = 0
    peak_memory_usage_mb: float = 0
    last_update: datetime = Field(default_factory=datetime.now)

    def update_processing_time(self, processing_time: float) -> None:
        """Update processing time statistics."""
        if self.chunks_processed == 0:
            self.avg_processing_time = processing_time
        else:
            # Moving average
            self.avg_processing_time = (
                (self.avg_processing_time * self.chunks_processed + processing_time) / 
                (self.chunks_processed + 1)
            )
        
        self.total_processing_time += processing_time
        self.chunks_processed += 1
        self.last_update = datetime.now()

    def log_error(self) -> None:
        """Log an error occurrence."""
        self.errors += 1
        self.last_update = datetime.now()

class PipelineManager:
    """
    Manages multiple video processing pipelines.
    """
    
    def __init__(self, 
                 base_data_dir: str = settings.PIPELINE.BASE_DATA_DIR, 
                 max_concurrent_pipelines: int = settings.PIPELINE.MAX_CONCURRENT_PIPELINES,
                 api_key: Optional[str] = None):
        """
        Initialize the pipeline manager.
        
        Args:
            base_data_dir: Base directory for storing pipeline data
            max_concurrent_pipelines: Maximum number of concurrent pipelines
            api_key: API key for analysis services
        """
        self.base_data_dir = base_data_dir
        self.max_concurrent_pipelines = max_concurrent_pipelines
        self.api_key = api_key or settings.ANALYSIS.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        
        # Pipelines dictionary: pipeline_id -> pipeline objects
        self.pipelines: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self.tasks: Dict[str, asyncio.Task] = {}
        
        # Processing queues (for when we exceed max_concurrent_pipelines)
        self.pending_queue: List[str] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Locks for thread safety
        self._pipeline_lock = asyncio.Lock()
        
    async def register_source(self, url: str, source_type: str = "youtube", 
                            metadata: Optional[Dict[str, Any]] = None) -> StreamSource:
        """
        Register a new streaming source.
        
        Args:
            url: Stream URL
            source_type: Type of source ("youtube", "tv", etc.)
            metadata: Additional metadata
            
        Returns:
            The created StreamSource object
        """
        source_id = str(uuid.uuid4())
        source = StreamSource(
            source_id=source_id, 
            url=url, 
            source_type=source_type, 
            metadata=metadata or {}
        )
        
        # If this is a new source, prepare its directory
        source_dir = os.path.join(self.base_data_dir, source_id)
        os.makedirs(source_dir, exist_ok=True)
        
        self.logger.info(f"Registered new source: {source_id} ({source_type}) - {url}")
        
        return source
        
    async def create_pipeline(self, source: StreamSource, 
                            chunk_duration: int = settings.PIPELINE.DEFAULT_CHUNK_DURATION,
                            analysis_prompt: Optional[str] = None,
                            use_web_search: bool = settings.ANALYSIS.USE_WEB_SEARCH,
                            export_responses: Optional[bool] = None) -> str:
        """
        Create a new pipeline for a source.
        
        Args:
            source: The source to process
            chunk_duration: Duration of each chunk in seconds
            analysis_prompt: Prompt for analysis
            use_web_search: Whether to use web search
            export_responses: Whether to export full responses to files (overrides settings)
            
        Returns:
            Pipeline ID
        """
        async with self._pipeline_lock:
            # Create a unique pipeline ID
            pipeline_id = str(uuid.uuid4())
            
            # Initialize components
            chunker = YouTubeChunker(base_data_folder=self.base_data_dir)
            
            # Determine if we should export responses
            should_export = export_responses if export_responses is not None else settings.ANALYSIS.EXPORT_RESPONSES
            is_test_mode = os.getenv("GEMINI_TEST_MODE", "").lower() == "true"
            
            # Initialize the analyzer with additional parameters
            analyzer = GeminiVideoAnalyzer(
                api_key=self.api_key,
                max_retries=settings.ANALYSIS.MAX_RETRIES,
                retry_delay=settings.ANALYSIS.RETRY_DELAY,
                enable_caching=True,
                base_data_dir=self.base_data_dir,
                export_responses=should_export and not is_test_mode
            )
            
            # Create state machine
            state_machine = PipelineStateMachine(pipeline_id, self.logger)
            
            # Setup pipeline data structure
            now = datetime.now()
            pipeline_dir = os.path.join(
                self.base_data_dir,
                source.source_id,
                now.strftime('%Y_%m_%d'),
                now.strftime('%H_%M')
            )
            os.makedirs(pipeline_dir, exist_ok=True)
            
            pipeline_data = {
                "pipeline_id": pipeline_id,
                "source": source,
                "created_at": now,
                "updated_at": now,
                "chunk_duration": chunk_duration,
                "analysis_prompt": analysis_prompt or settings.ANALYSIS.DEFAULT_ANALYSIS_PROMPT,
                "use_web_search": use_web_search,
                "output_dir": pipeline_dir,
                "chunker": chunker,
                "analyzer": analyzer,
                "state_machine": state_machine,
                "processed_chunks": set(),
                "chunk_queue": asyncio.Queue(),
                "stats": PipelineStats()
            }
            
            # Store pipeline data
            self.pipelines[pipeline_id] = pipeline_data
            
            self.logger.info(f"Created pipeline {pipeline_id} for source {source.source_id}")
            
            return pipeline_id
            
    async def start_pipeline(self, pipeline_id: str) -> bool:
        """
        Start a pipeline.
        
        Args:
            pipeline_id: ID of the pipeline to start
            
        Returns:
            True if started successfully, False otherwise
        """
        if pipeline_id not in self.pipelines:
            self.logger.error(f"Cannot start pipeline {pipeline_id}: not found")
            return False
            
        pipeline = self.pipelines[pipeline_id]
        state_machine = pipeline["state_machine"]
        
        # Check if the pipeline can be started
        if not state_machine.can_transition_to(PipelineState.STARTING):
            self.logger.warning(f"Cannot start pipeline {pipeline_id} in state {state_machine.get_current_state()}")
            return False
            
        # Check if we have capacity
        active_count = sum(1 for p in self.pipelines.values() 
                          if p["state_machine"].is_active())
                          
        if active_count >= self.max_concurrent_pipelines:
            # Add to pending queue
            self.logger.info(f"Pipeline {pipeline_id} added to pending queue (active: {active_count})")
            self.pending_queue.append(pipeline_id)
            return True
            
        # Transition state
        if not await state_machine.transition_to(PipelineState.STARTING):
            self.logger.error(f"Failed to transition pipeline {pipeline_id} to STARTING state")
            return False
            
        # Create and start task
        task = asyncio.create_task(self._run_pipeline(pipeline_id))
        self.tasks[pipeline_id] = task
        
        self.logger.info(f"Started pipeline {pipeline_id}")
        
        return True
        
    async def stop_pipeline(self, pipeline_id: str) -> bool:
        """
        Stop a pipeline.
        
        Args:
            pipeline_id: ID of the pipeline to stop
            
        Returns:
            True if stop signal sent successfully, False otherwise
        """
        if pipeline_id not in self.pipelines:
            self.logger.error(f"Cannot stop pipeline {pipeline_id}: not found")
            return False
            
        pipeline = self.pipelines[pipeline_id]
        state_machine = pipeline["state_machine"]
        
        # Check if the pipeline can be stopped
        if not state_machine.can_transition_to(PipelineState.STOPPING):
            self.logger.warning(f"Cannot stop pipeline {pipeline_id} in state {state_machine.get_current_state()}")
            
            # If it's in the pending queue, just remove it
            if pipeline_id in self.pending_queue:
                self.pending_queue.remove(pipeline_id)
                await state_machine.transition_to(PipelineState.STOPPED)
                self.logger.info(f"Removed pipeline {pipeline_id} from pending queue")
                return True
                
            return False
            
        # Transition state
        if not await state_machine.transition_to(PipelineState.STOPPING):
            self.logger.error(f"Failed to transition pipeline {pipeline_id} to STOPPING state")
            return False
            
        self.logger.info(f"Stop signal sent to pipeline {pipeline_id}")
        
        return True
        
    async def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a pipeline.
        
        Args:
            pipeline_id: ID of the pipeline
            
        Returns:
            Dictionary with pipeline status or None if not found
        """
        if pipeline_id not in self.pipelines:
            return None
            
        pipeline = self.pipelines[pipeline_id]
        state_machine = pipeline["state_machine"]
        source = pipeline["source"]
        
        status = {
            "pipeline_id": pipeline_id,
            "source_id": source.source_id,
            "url": source.url,
            "state": state_machine.get_current_state().value,  # Get string value from enum
            "state_duration": state_machine.get_state_duration(),
            "created_at": pipeline["created_at"].isoformat(),
            "updated_at": pipeline["updated_at"].isoformat(),
            "stats": pipeline["stats"].model_dump(),
            "output_dir": pipeline["output_dir"]
        }
        
        return status
        
    async def get_all_pipeline_statuses(self) -> List[Dict[str, Any]]:
        """
        Get status of all pipelines.
        
        Returns:
            List of pipeline status dictionaries
        """
        return [await self.get_pipeline_status(pipeline_id) for pipeline_id in self.pipelines]
        
    async def _run_pipeline(self, pipeline_id: str) -> None:
        """
        Run the pipeline processing logic.
        
        Args:
            pipeline_id: ID of the pipeline to run
        """
        if pipeline_id not in self.pipelines:
            self.logger.error(f"Cannot run pipeline {pipeline_id}: not found")
            return
            
        pipeline = self.pipelines[pipeline_id]
        state_machine = pipeline["state_machine"]
        source = pipeline["source"]
        
        chunker = pipeline["chunker"]
        analyzer = pipeline["analyzer"]
        chunk_duration = pipeline["chunk_duration"]
        analysis_prompt = pipeline["analysis_prompt"]
        use_web_search = pipeline["use_web_search"]
        output_dir = pipeline["output_dir"]
        chunk_queue = pipeline["chunk_queue"]
        processed_chunks = pipeline["processed_chunks"]
        
        # Setup file system monitoring
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class ChunkHandler(FileSystemEventHandler):
            def __init__(self, queue, loop):
                self.queue = queue
                self.loop = loop
                
            def on_created(self, event):
                if not event.is_directory and event.src_path.endswith('.mp4'):
                    asyncio.run_coroutine_threadsafe(
                        self.queue.put(event.src_path),
                        self.loop
                    )
        
        # Start the observer
        observer = Observer()
        event_handler = ChunkHandler(chunk_queue, asyncio.get_running_loop())
        observer.schedule(event_handler, output_dir, recursive=False)
        observer.start()
        
        try:
            # Transition to RUNNING state
            if not await state_machine.transition_to(PipelineState.RUNNING):
                self.logger.error(f"Failed to transition pipeline {pipeline_id} to RUNNING state")
                return
                
            # Start chunking task
            chunking_task = asyncio.create_task(self._run_chunking(
                pipeline_id, source.url, chunk_duration, output_dir, chunker
            ))
            
            # Start analysis task
            analysis_task = asyncio.create_task(self._run_analysis(
                pipeline_id, chunk_queue, processed_chunks, analyzer, 
                analysis_prompt, use_web_search
            ))
            
            # Wait for both tasks to complete or for stop signal
            done, pending = await asyncio.wait(
                [chunking_task, analysis_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Check if state machine was stopped externally
            if state_machine.get_current_state() == PipelineState.STOPPING:
                for task in pending:
                    task.cancel()
                    
                # Wait for tasks to cancel
                try:
                    await asyncio.gather(*pending, return_exceptions=True)
                except asyncio.CancelledError:
                    pass
                    
                # Transition to STOPPED
                await state_machine.transition_to(PipelineState.STOPPED)
            else:
                # Check if any task failed
                for task in done:
                    try:
                        exception = task.exception()
                        if exception:
                            self.logger.error(f"Pipeline {pipeline_id} task failed: {exception}")
                            await state_machine.transition_to(PipelineState.FAILED, {
                                "error": str(exception)
                            })
                            
                            # Cancel remaining tasks
                            for t in pending:
                                t.cancel()
                            break
                    except asyncio.CancelledError:
                        pass
                
                # If we get here and state is still RUNNING, transition to COMPLETED
                if state_machine.get_current_state() == PipelineState.RUNNING:
                    await state_machine.transition_to(PipelineState.COMPLETED)
        
        except Exception as e:
            self.logger.error(f"Pipeline {pipeline_id} execution error: {e}")
            await state_machine.transition_to(PipelineState.FAILED, {
                "error": str(e)
            })
        finally:
            # Stop the observer
            observer.stop()
            observer.join()
            
            # Remove from active tasks
            if pipeline_id in self.tasks:
                del self.tasks[pipeline_id]
                
            # Update pipeline statistics
            pipeline["updated_at"] = datetime.now()
            
            # Process the next pipeline in the queue if there is one
            if self.pending_queue:
                next_pipeline_id = self.pending_queue.pop(0)
                await self.start_pipeline(next_pipeline_id)
                
    async def _run_chunking(self, pipeline_id: str, url: str, chunk_duration: int, 
                           output_dir: str, chunker: YouTubeChunker) -> None:
        """Run the chunking process for a pipeline."""
        pipeline = self.pipelines[pipeline_id]
        state_machine = pipeline["state_machine"]
        
        try:
            self.logger.info(f"Starting chunking for pipeline {pipeline_id}")
            
            # Only continue if we're in RUNNING state
            if state_machine.get_current_state() != PipelineState.RUNNING:
                return
                
            # Start chunking
            await asyncio.to_thread(
                chunker.process_video,
                url=url,
                chunk_duration=chunk_duration,
                output_dir=output_dir
            )
            
            self.logger.info(f"Chunking completed for pipeline {pipeline_id}")
            
            # Signal end of chunking by putting None in the queue
            await pipeline["chunk_queue"].put(None)
            
        except Exception as e:
            self.logger.error(f"Chunking error for pipeline {pipeline_id}: {e}")
            # Signal error to the analysis task
            await pipeline["chunk_queue"].put(None)
            raise
            
    async def _run_analysis(self, pipeline_id: str, chunk_queue: asyncio.Queue,
                       processed_chunks: Set[str], analyzer: GeminiVideoAnalyzer,
                       analysis_prompt: str, use_web_search: bool) -> None:
        """Run the analysis process for a pipeline."""
        pipeline = self.pipelines[pipeline_id]
        state_machine = pipeline["state_machine"]
        stats = pipeline["stats"]
        chunk_duration = pipeline["chunk_duration"]
        safety_margin = 1  # Additional second to ensure chunk is completely written
        
        try:
            self.logger.info(f"Starting analysis for pipeline {pipeline_id}")
            
            from ..api.websocket_manager import manager
            
            while True:
                # Check if pipeline is stopping
                if state_machine.get_current_state() in [PipelineState.STOPPING, PipelineState.STOPPED]:
                    self.logger.info(f"Stopping analysis for pipeline {pipeline_id}")
                    break
                    
                try:
                    # Get the next chunk with timeout to periodically check state
                    chunk_path = await asyncio.wait_for(
                        chunk_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # No chunk available, continue checking state
                    continue
                    
                # Check for end signal
                if chunk_path is None:
                    self.logger.info(f"End of chunking for pipeline {pipeline_id}")
                    break
                    
                # Skip if already processed
                if chunk_path in processed_chunks:
                    self.logger.info(f"Chunk already processed: {chunk_path}")
                    chunk_queue.task_done()
                    continue
                    
                # IMPORTANT: Wait for chunk to be fully written
                self.logger.info(f"Waiting {chunk_duration + safety_margin}s for chunk to complete: {os.path.basename(chunk_path)}")
                await asyncio.sleep(chunk_duration + safety_margin)
                
                # Verify file still exists and check size
                if not os.path.exists(chunk_path):
                    self.logger.warning(f"Chunk file disappeared during waiting period: {chunk_path}")
                    chunk_queue.task_done()
                    continue
                    
                file_size = os.path.getsize(chunk_path)
                self.logger.info(f"Chunk size after waiting: {file_size} bytes for {os.path.basename(chunk_path)}")
                
                if file_size < 1024 and not os.getenv("GEMINI_TEST_MODE", "").lower() == "true":  # Less than 1KB
                    self.logger.warning(f"Chunk file too small ({file_size} bytes) after waiting: {chunk_path}")
                    chunk_queue.task_done()
                    continue
                
                # Analyze the chunk
                self.logger.info(f"Analyzing chunk: {chunk_path}")
                
                try:
                    start_time = time.time()
                    
                    # Use web search if configured - run in a separate thread
                    if use_web_search:
                        result = await asyncio.to_thread(
                            analyzer.analyze_video_with_web_search,
                            video_path=chunk_path,
                            prompt=analysis_prompt
                        )
                    else:
                        result = await asyncio.to_thread(
                            analyzer.analyze_video,
                            video_path=chunk_path,
                            prompt=analysis_prompt
                        )
                    
                    processing_time = time.time() - start_time
                        
                    # Broadcast result
                    await manager.broadcast_analysis(result, chunk_path)
                    
                    # Update pipeline stats
                    stats.update_processing_time(processing_time)
                    processed_chunks.add(chunk_path)
                    
                    self.logger.info(f"Analysis complete for chunk: {chunk_path} in {processing_time:.2f}s")
                except Exception as e:
                    stats.log_error()
                    self.logger.error(f"Analysis error for chunk {chunk_path}: {e}")
                    
                    # Broadcast error
                    await manager.broadcast_analysis(
                        f"Error analyzing chunk: {str(e)[:100]}...", 
                        chunk_path
                    )
                finally:
                    # Mark task as done
                    chunk_queue.task_done()
        
        except Exception as e:
            self.logger.error(f"Analysis process error for pipeline {pipeline_id}: {e}")
            raise