import asyncio
import os
import logging
import time
from datetime import datetime
from typing import Optional, Set, Dict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ChunkProcessingHandler(FileSystemEventHandler):
    """Handles file system events for new video chunks."""
    
    def __init__(self, chunk_queue: asyncio.Queue, processed_chunks: Set[str], 
                 chunk_timestamps: Dict[str, float], loop: asyncio.AbstractEventLoop):
        """
        Initialize the handler.
        
        Args:
            chunk_queue: Queue for new chunks
            processed_chunks: Set of processed chunk paths
            chunk_timestamps: Dictionary to store chunk creation times
            loop: Event loop to use for async operations
        """
        self.chunk_queue = chunk_queue
        self.processed_chunks = processed_chunks
        self.chunk_timestamps = chunk_timestamps
        self.loop = loop
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.mp4'):
            # Store creation timestamp
            self.chunk_timestamps[event.src_path] = time.time()
            
            # Queue the chunk
            future = asyncio.run_coroutine_threadsafe(
                self.chunk_queue.put(event.src_path),
                self.loop
            )
            try:
                future.result(timeout=1.0)
                self.logger.info(f"New chunk queued: {os.path.basename(event.src_path)}")
            except Exception as e:
                self.logger.error(f"Error queueing chunk {event.src_path}: {e}")

class VideoPipeline:
    """Coordinates video chunking and analysis processes."""
    
    def __init__(self, chunker, analyzer, chunk_dir: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            chunker: YouTubeChunker instance
            analyzer: GeminiVideoAnalyzer instance
            chunk_dir: Optional directory to store chunks. If None, creates based on date/time
        """
        self.chunker = chunker
        self.analyzer = analyzer
        self.chunk_dir = chunk_dir
        self.chunk_queue = asyncio.Queue()
        self.processed_chunks = set()
        self.chunk_timestamps = {}
        self.is_running = False
        self.observer = None
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def start_pipeline(self, url: str, chunk_duration: int, analysis_prompt: str, use_web_search: bool = False):
        """
        Start the video processing pipeline.
        
        Args:
            url: YouTube URL to process
            chunk_duration: Duration of each chunk in seconds
            analysis_prompt: Prompt for Gemini analysis
            use_web_search: Whether to use web search in analysis
        """
        self.observer = Observer()
        self.chunk_duration = chunk_duration
        
        try:
            # Start chunking process
            self.is_running = True
            self.use_web_search = use_web_search
            
            # Get the current event loop
            loop = asyncio.get_running_loop()
            
            # If no chunk_dir specified, create one based on current date/time
            if not self.chunk_dir:
                now = datetime.now()
                self.chunk_dir = os.path.join(
                    'data',
                    now.strftime('%Y_%m_%d'),
                    now.strftime('%H_%M')
                )
            
            # Ensure directory exists
            os.makedirs(self.chunk_dir, exist_ok=True)
            
            # Set up file system monitoring
            event_handler = ChunkProcessingHandler(
                self.chunk_queue,
                self.processed_chunks,
                self.chunk_timestamps,
                loop
            )
            
            # Start monitoring for new chunks
            self.observer.schedule(event_handler, self.chunk_dir, recursive=False)
            self.observer.start()
            self.logger.info(f"Started file monitoring in {self.chunk_dir}")
            
            # Start chunking and analysis as concurrent tasks
            tasks = [
                asyncio.create_task(self._run_chunking(url, chunk_duration)),
                asyncio.create_task(self._run_analysis(analysis_prompt))
            ]
            
            # Wait for both tasks or handle cancellation
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                self.logger.info("Received cancellation request")
                for task in tasks:
                    if not task.done():
                        task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
            except Exception as e:
                self.logger.error(f"Error in tasks: {e}")
                raise
                
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            raise
        finally:
            await self._cleanup()

    async def _cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up pipeline...")
        self.is_running = False
        
        # Stop the observer if it was started
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
        
        # Signal analysis task to stop
        await self.chunk_queue.put(None)
        
        self.logger.info("Pipeline cleanup complete")

    async def _run_chunking(self, url: str, chunk_duration: int):
        """Run the video chunking process."""
        try:
            self.logger.info("Starting video chunking...")
            await asyncio.to_thread(
                self.chunker.process_video,
                url=url,
                chunk_duration=chunk_duration,
                output_dir=self.chunk_dir
            )
        except Exception as e:
            self.logger.error(f"Error in chunking process: {e}")
            raise
        finally:
            # Signal end of chunking
            await self.chunk_queue.put(None)

    async def _run_analysis(self, prompt: str):
        """Run the analysis process for each chunk."""
        self.logger.info("Starting analysis process...")
        safety_margin = 1  # Additional seconds to wait after chunk duration
        
        while self.is_running:
            try:
                # Get next chunk from queue with timeout
                try:
                    chunk_path = await asyncio.wait_for(
                        self.chunk_queue.get(),
                        timeout=1.0  # 1 second timeout
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check for stop signal
                if chunk_path is None:
                    self.logger.info("Received stop signal for analysis")
                    break
                    
                try:
                    # Skip if already processed
                    if chunk_path in self.processed_chunks:
                        continue
                        
                    # Get chunk creation time
                    creation_time = self.chunk_timestamps.get(chunk_path, time.time())
                    elapsed = time.time() - creation_time
                    wait_time = max(0, (self.chunk_duration + safety_margin) - elapsed)
                    
                    if wait_time > 0:
                        self.logger.info(f"Waiting {wait_time:.1f} seconds for chunk to complete: {os.path.basename(chunk_path)}")
                        await asyncio.sleep(wait_time)
                    
                    # Verify file exists and has minimum size
                    if not os.path.exists(chunk_path):
                        self.logger.warning(f"Chunk file disappeared: {chunk_path}")
                        continue
                        
                    file_size = os.path.getsize(chunk_path)
                    if file_size < 1024:  # Less than 1KB
                        self.logger.warning(f"Chunk file too small ({file_size} bytes): {chunk_path}")
                        continue
                    
                    # Analyze chunk
                    self.logger.info(f"Analyzing chunk: {os.path.basename(chunk_path)}")
                    
                    if self.use_web_search:
                        result = await asyncio.to_thread(
                            self.analyzer.analyze_video_with_web_search,
                            video_path=chunk_path,
                            prompt=prompt
                        )
                    else:
                        result = await asyncio.to_thread(
                            self.analyzer.analyze_video,
                            video_path=chunk_path,
                            prompt=prompt
                        )
                    
                    # Mark as processed and clean up timestamp
                    self.processed_chunks.add(chunk_path)
                    self.chunk_timestamps.pop(chunk_path, None)
                    
                    # Log result
                    self.logger.info(f"Analysis result for {os.path.basename(chunk_path)}: {result}")
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing chunk {chunk_path}: {e}")
                finally:
                    self.chunk_queue.task_done()
                    
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                if not self.is_running:
                    break

# Example usage:
if __name__ == "__main__":
    import os
    from src.recorder.stream_recorder import YouTubeChunker
    from src.analyzer.gemini_analyzer import GeminiVideoAnalyzer
    
    async def main():
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please set GOOGLE_API_KEY environment variable")
            
        # Initialize components
        chunker = YouTubeChunker()
        analyzer = GeminiVideoAnalyzer(api_key=api_key)
        
        # Create pipeline
        pipeline = VideoPipeline(chunker, analyzer)
        
        try:
            # Start pipeline
            await pipeline.start_pipeline(
                url="https://www.youtube.com/watch?v=example",
                chunk_duration=15,
                analysis_prompt="Describe what happens in this video segment",
                use_web_search=False
            )
        except KeyboardInterrupt:
            pass  # Cleanup will happen in finally block

    # Run the pipeline
    asyncio.run(main())