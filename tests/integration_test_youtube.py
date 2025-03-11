# integration_test.py
"""
Integration test for the video analysis pipeline.

This script tests the core components of the video analysis pipeline:
- WebSocket Manager
- Pipeline Manager
- Stream Recorder
- Gemini Analyzer

Usage:
    python integration_test.py [youtube_url]

If no URL is provided, a default livestream URL will be used.
For best results, use a short video (under 1 minute) or a livestream.
"""

import os
import asyncio
import logging
import time
import sys
from datetime import datetime
from dotenv import load_dotenv

from src.core.config import settings, initialize_logger
from src.core.state_machine import PipelineState
from src.core.pipeline_manager import PipelineManager, StreamSource
from src.recorder.youtube_recorder import YouTubeChunker
from src.analyzer.gemini_analyzer import GeminiVideoAnalyzer
from src.api.websocket_manager import manager

os.environ["GEMINI_TEST_MODE"] = "false"
load_dotenv()

# Setup logging
logger = initialize_logger()

async def test_websocket_broadcast():
    """Test WebSocket broadcasting."""
    logger.info("Testing WebSocket broadcasting...")
    
    # Broadcast a test message
    await manager.broadcast_message({
        "type": "test",
        "message": "Test message",
        "timestamp": datetime.now().isoformat()
    })
    
    logger.info("WebSocket broadcast test complete")
    return True

async def test_pipeline_manager(url: str):
    """
    Test the pipeline manager with the provided URL.
    
    Args:
        url: YouTube URL to use for testing
    """
    logger.info(f"Testing pipeline manager with URL: {url}")
    
    # Create pipeline manager
    pipeline_manager = PipelineManager(
        base_data_dir=settings.PIPELINE.BASE_DATA_DIR,
        max_concurrent_pipelines=settings.PIPELINE.MAX_CONCURRENT_PIPELINES,
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Register the test source
    source = await pipeline_manager.register_source(
        url=url,
        source_type="youtube",
        metadata={"test": True}
    )
    
    logger.info(f"Registered source: {source.source_id}")
    
    # Create pipeline
    pipeline_id = await pipeline_manager.create_pipeline(
        source=source,
        chunk_duration=30,  # 10-second chunks for quicker testing
        analysis_prompt="Describe this video clip briefly. What do you see?",
        use_web_search=False  # Faster without web search
    )
    
    logger.info(f"Created pipeline: {pipeline_id}")
    
    # Start pipeline
    success = await pipeline_manager.start_pipeline(pipeline_id)
    if not success:
        logger.error("Failed to start pipeline")
        return False
        
    logger.info(f"Started pipeline: {pipeline_id}")
    
    # Monitor pipeline state
    max_wait_time = 300  # 3 minutes max
    start_time = time.time()
    completed_states = {PipelineState.COMPLETED, PipelineState.FAILED, PipelineState.STOPPED}
    
    while time.time() - start_time < max_wait_time:
        # Get current status
        status = await pipeline_manager.get_pipeline_status(pipeline_id)
        if not status:
            logger.error("Pipeline status not found")
            return False
            
        state = PipelineState(status["state"])
        chunks_processed = status["stats"]["chunks_processed"]
        logger.info(f"Pipeline state: {state.value}, Chunks processed: {chunks_processed}")
        
        # Success if we've processed at least one chunk
        if chunks_processed > 0:
            logger.info(f"Successfully processed {chunks_processed} chunks")
            
            # Stop the pipeline after processing at least one chunk
            await pipeline_manager.stop_pipeline(pipeline_id)
            return True
            
        # Check if in a terminal state with no chunks processed
        if state in completed_states and chunks_processed == 0:
            logger.error(f"Pipeline reached terminal state {state.value} with no chunks processed")
            return False
            
        # Wait before checking again
        await asyncio.sleep(5)
    
    # Timeout - try to stop the pipeline
    logger.warning("Timed out waiting for pipeline to process chunks")
    await pipeline_manager.stop_pipeline(pipeline_id)
    
    # Now verify that chunking has actually stopped
    chunking_stopped = await verify_chunking_stopped(pipeline_manager, pipeline_id)
    if not chunking_stopped:
        logger.error("Pipeline chunking did not stop correctly")
        return False

async def test_error_handling():
    """Test error handling with an invalid URL."""
    logger.info("Testing error handling...")
    
    # Create pipeline manager
    pipeline_manager = PipelineManager(
        base_data_dir=settings.PIPELINE.BASE_DATA_DIR,
        max_concurrent_pipelines=settings.PIPELINE.MAX_CONCURRENT_PIPELINES,
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Register a test source with invalid URL
    test_url = "https://www.youtube.com/watch?v=invalid_video_id_that_doesnt_exist"
    
    source = await pipeline_manager.register_source(
        url=test_url,
        source_type="youtube",
        metadata={"test": True}
    )
    
    logger.info(f"Registered source with invalid URL: {source.source_id}")
    
    # Create pipeline
    pipeline_id = await pipeline_manager.create_pipeline(
        source=source,
        chunk_duration=10,
        analysis_prompt="Describe this video briefly",
        use_web_search=False
    )
    
    logger.info(f"Created pipeline: {pipeline_id}")
    
    # Start pipeline
    success = await pipeline_manager.start_pipeline(pipeline_id)
    if not success:
        logger.error("Failed to start pipeline")
        return False
        
    logger.info(f"Started pipeline with invalid URL: {pipeline_id}")
    
    # Monitor pipeline state
    max_wait_time = 60  # 1 minute max
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        # Get current status
        status = await pipeline_manager.get_pipeline_status(pipeline_id)
        if not status:
            logger.error("Pipeline status not found")
            return False
            
        state = PipelineState(status["state"])
        logger.info(f"Pipeline state: {state.value}")
        
        # We expect the pipeline to fail
        if state == PipelineState.FAILED:
            logger.info("Pipeline failed as expected")
            return True
            
        # Wait before checking again
        await asyncio.sleep(5)
    
    # Check if we timed out
    if time.time() - start_time >= max_wait_time:
        logger.warning("Timed out waiting for pipeline to fail")
        # Try to stop the pipeline
        await pipeline_manager.stop_pipeline(pipeline_id)
        return False
    
    return False

async def test_multiple_pipelines(url: str):
    """
    Test running multiple pipelines concurrently using the same URL.
    
    Args:
        url: YouTube URL to use for testing
    """
    logger.info(f"Testing multiple pipelines with URL: {url}")
    
    # Create pipeline manager
    pipeline_manager = PipelineManager(
        base_data_dir=settings.PIPELINE.BASE_DATA_DIR,
        max_concurrent_pipelines=2,  # Limit to 2 concurrent pipelines
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Create 3 pipelines with different chunk durations
    pipeline_ids = []
    chunk_durations = [30, 35, 40]  # Different chunk durations test 5,10,15 in dummy test
    
    for i, duration in enumerate(chunk_durations):
        # Register source
        source = await pipeline_manager.register_source(
            url=url,
            source_type="youtube",
            metadata={"test": True, "index": i}
        )
        
        # Create pipeline
        pipeline_id = await pipeline_manager.create_pipeline(
            source=source,
            chunk_duration=duration,
            analysis_prompt=f"Describe this video clip with {duration}s chunks briefly. What do you see?",
            use_web_search=False
        )
        
        pipeline_ids.append(pipeline_id)
        logger.info(f"Created pipeline {i+1}: {pipeline_id} with {duration}s chunks")
        
        # Start pipeline
        success = await pipeline_manager.start_pipeline(pipeline_id)
        if success:
            logger.info(f"Started pipeline {i+1}: {pipeline_id}")
        else:
            logger.info(f"Pipeline {i+1} queued for later execution")
    
    # Monitor pipelines
    max_wait_time = 180  # 3 minutes max
    start_time = time.time()
    processed_chunks = 0
    
    while time.time() - start_time < max_wait_time and processed_chunks < 2:  # Wait for at least 2 chunks
        # Check all pipelines
        active_count = 0
        processed_chunks = 0
        
        for pipeline_id in pipeline_ids:
            status = await pipeline_manager.get_pipeline_status(pipeline_id)
            if not status:
                continue
                
            state = PipelineState(status["state"])
            chunks = status["stats"]["chunks_processed"]
            processed_chunks += chunks
            
            if state.value in {"running", "starting", "pausing"}:
                active_count += 1
        
        logger.info(f"Active pipelines: {active_count}, Total chunks processed: {processed_chunks}")
        
        # If we have processed enough chunks, we can stop
        if processed_chunks >= 2:
            break
            
        # Wait before checking again
        await asyncio.sleep(5)
    
    # Stop all pipelines
    for pipeline_id in pipeline_ids:
        await pipeline_manager.stop_pipeline(pipeline_id)
    
    # Check results
    all_statuses = await pipeline_manager.get_all_pipeline_statuses()
    logger.info(f"Final pipeline statuses: {all_statuses}")
    
    # Success if we processed at least 2 chunks total across all pipelines
    success = processed_chunks >= 2
    
    if success:
        logger.info(f"Multiple pipeline test completed successfully with {processed_chunks} chunks processed")
    else:
        logger.error(f"Multiple pipeline test failed - only {processed_chunks} chunks processed")
        
    return success

async def verify_chunking_stopped(pipeline_manager, pipeline_id, max_wait=30):
    """Verify the chunking process has truly stopped."""
    logger.info("Verifying chunking processes have stopped...")
    
    status = await pipeline_manager.get_pipeline_status(pipeline_id)
    if not status:
        logger.error("Pipeline status not found")
        return False
        
    # Get the output directory
    output_dir = status["output_dir"]
    
    # Check initial file count
    initial_files = set(os.listdir(output_dir))
    initial_count = len([f for f in initial_files if f.endswith('.mp4')])
    
    logger.info(f"Initial chunk count: {initial_count}")
    
    # Wait and check if new files appear
    wait_time = 5  # Wait 5 seconds between checks
    for i in range(max_wait // wait_time):
        await asyncio.sleep(wait_time)
        
        current_files = set(os.listdir(output_dir))
        current_count = len([f for f in current_files if f.endswith('.mp4')])
        
        new_files = current_files - initial_files
        new_chunks = [f for f in new_files if f.endswith('.mp4')]
        
        if new_chunks:
            logger.warning(f"New chunks detected after stop: {new_chunks}")
            return False
            
        logger.info(f"Chunk count after {(i+1)*wait_time}s: {current_count} (no new chunks)")
        
    logger.info("No new chunks detected after stop, chunking has stopped successfully")
    return True

async def main():
    """Run all integration tests with a single YouTube URL."""
    try:
        # Default to a popular live stream if no URL is provided
        default_url = "https://www.youtube.com/watch?v=LPmbtKSwN6E" 
        
        # Check if URL is provided as command line argument
        if len(sys.argv) > 1:
            youtube_url = sys.argv[1]
        else:
            youtube_url = default_url
            
        logger.info(f"Starting integration tests with URL: {youtube_url}")
        
        # Run tests
        websocket_test = await test_websocket_broadcast()
        logger.info(f"WebSocket test result: {'PASS' if websocket_test else 'FAIL'}")
        
        pipeline_test = await test_pipeline_manager(youtube_url)
        logger.info(f"Pipeline test result: {'PASS' if pipeline_test else 'FAIL'}")
        
        error_test = await test_error_handling()
        logger.info(f"Error handling test result: {'PASS' if error_test else 'FAIL'}")
        
        multiple_test = await test_multiple_pipelines(youtube_url)
        logger.info(f"Multiple pipeline test result: {'PASS' if multiple_test else 'FAIL'}")
        
        # Overall result
        all_passed = websocket_test and pipeline_test and error_test and multiple_test
        logger.info(f"Integration tests overall result: {'PASS' if all_passed else 'FAIL'}")
        
        return all_passed
        
    except Exception as e:
        logger.error(f"Error in integration tests: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Run the integration tests
    result = asyncio.run(main())
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if result else 1)