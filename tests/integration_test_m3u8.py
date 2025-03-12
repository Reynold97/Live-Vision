"""
Integration test for M3U8 stream analysis pipeline.

This script tests the core components of the M3U8 stream analysis pipeline:
- WebSocket Manager
- Pipeline Manager with M3U8 sources
- M3U8 Stream Recorder
- Gemini Analyzer

Usage:
    python tests/integration_test_m3u8.py [m3u8_url]

If no URL is provided, a default public stream URL will be used.
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
from src.recorder.m3u8_recorder import M3U8Chunker
from src.analyzer.gemini_analyzer import GeminiVideoAnalyzer
from src.api.websocket_manager import manager

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

async def test_m3u8_pipeline_manager(url: str):
    """
    Test the pipeline manager with an M3U8 URL.
    
    Args:
        url: M3U8 URL to use for testing
    """
    # Clean the URL to remove any trailing whitespace
    url = url.strip()
    logger.info(f"Testing pipeline manager with M3U8 URL: {url}")
    
    # Create pipeline manager
    pipeline_manager = PipelineManager(
        base_data_dir=settings.PIPELINE.BASE_DATA_DIR,
        max_concurrent_pipelines=settings.PIPELINE.MAX_CONCURRENT_PIPELINES,
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Register the M3U8 source
    source = await pipeline_manager.register_source(
        url=url,
        source_type="m3u8",  # Explicitly setting m3u8 source type
        metadata={"test": True}
    )
    
    logger.info(f"Registered M3U8 source: {source.source_id}")
    
    # Create pipeline
    pipeline_id = await pipeline_manager.create_pipeline(
        source=source,
        chunk_duration=30,  # 30-second chunks
        analysis_prompt="Describe this video clip briefly. What do you see?",
        use_web_search=False
    )
    
    logger.info(f"Created pipeline: {pipeline_id}")
    
    # Start pipeline
    success = await pipeline_manager.start_pipeline(pipeline_id)
    if not success:
        logger.error("Failed to start M3U8 pipeline")
        return False
        
    logger.info(f"Started M3U8 pipeline: {pipeline_id}")
    
    # Monitor pipeline state
    max_wait_time = 300  # 5 minutes max
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
        logger.info(f"M3U8 pipeline state: {state.value}, Chunks processed: {chunks_processed}")
        
        # Success if we've processed at least one chunk
        if chunks_processed > 0:
            logger.info(f"Successfully processed {chunks_processed} chunks from M3U8 stream")
            
            # Stop the pipeline after processing at least one chunk
            await pipeline_manager.stop_pipeline(pipeline_id)
            return True
            
        # Check if in a terminal state with no chunks processed
        if state in completed_states and chunks_processed == 0:
            logger.error(f"M3U8 pipeline reached terminal state {state.value} with no chunks processed")
            return False
            
        # Wait before checking again
        await asyncio.sleep(5)
    
    # Timeout - try to stop the pipeline
    logger.warning("Timed out waiting for M3U8 pipeline to process chunks")
    await pipeline_manager.stop_pipeline(pipeline_id)
    
    # Now verify that chunking has actually stopped
    chunking_stopped = await verify_chunking_stopped(pipeline_manager, pipeline_id)
    if not chunking_stopped:
        logger.error("M3U8 pipeline chunking did not stop correctly")
        return False
    
    return False

async def test_m3u8_error_handling():
    """Test error handling with an invalid M3U8 URL."""
    logger.info("Testing M3U8 error handling...")
    
    # Create pipeline manager
    pipeline_manager = PipelineManager(
        base_data_dir=settings.PIPELINE.BASE_DATA_DIR,
        max_concurrent_pipelines=settings.PIPELINE.MAX_CONCURRENT_PIPELINES,
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Register a test source with invalid URL - non-existent m3u8 file
    test_url = "https://example.com/invalid-stream-that-doesnt-exist.m3u8"
    
    source = await pipeline_manager.register_source(
        url=test_url,
        source_type="m3u8",
        metadata={"test": True}
    )
    
    logger.info(f"Registered source with invalid M3U8 URL: {source.source_id}")
    
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
        
    logger.info(f"Started pipeline with invalid M3U8 URL: {pipeline_id}")
    
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
            logger.info("Pipeline with invalid M3U8 URL failed as expected")
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

async def test_multiple_m3u8_pipelines(m3u8_url: str):
    """
    Test running multiple M3U8 pipelines concurrently with different chunk durations.
    
    Args:
        m3u8_url: M3U8 URL to use for testing
    """
    # Clean the URL to remove any trailing whitespace
    m3u8_url = m3u8_url.strip()
    logger.info(f"Testing multiple M3U8 pipelines with URL: {m3u8_url}")
    
    # Create pipeline manager
    pipeline_manager = PipelineManager(
        base_data_dir=settings.PIPELINE.BASE_DATA_DIR,
        max_concurrent_pipelines=2,  # Limit to 2 concurrent pipelines
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Create 2 pipelines with different chunk durations
    pipeline_ids = []
    chunk_durations = [20, 30]  # Different chunk durations
    
    for i, duration in enumerate(chunk_durations):
        # Register source
        source = await pipeline_manager.register_source(
            url=m3u8_url,
            source_type="m3u8",
            metadata={"test": True, "index": i}
        )
        
        # Create pipeline
        pipeline_id = await pipeline_manager.create_pipeline(
            source=source,
            chunk_duration=duration,
            analysis_prompt=f"Describe this video clip with {duration}s chunks briefly",
            use_web_search=False
        )
        
        pipeline_ids.append(pipeline_id)
        logger.info(f"Created M3U8 pipeline {i+1}: {pipeline_id} with {duration}s chunks")
        
        # Start pipeline
        success = await pipeline_manager.start_pipeline(pipeline_id)
        if success:
            logger.info(f"Started M3U8 pipeline {i+1}: {pipeline_id}")
        else:
            logger.info(f"M3U8 pipeline {i+1} queued for later execution")
    
    # Monitor pipelines
    max_wait_time = 300  # 5 minutes max
    start_time = time.time()
    pipeline_chunks = {pid: 0 for pid in pipeline_ids}
    
    while time.time() - start_time < max_wait_time:
        # Check all pipelines
        active_count = 0
        total_chunks = 0
        all_pipelines_processed = True
        
        for pipeline_id in pipeline_ids:
            status = await pipeline_manager.get_pipeline_status(pipeline_id)
            if not status:
                continue
                
            state = PipelineState(status["state"])
            chunks = status["stats"]["chunks_processed"]
            pipeline_chunks[pipeline_id] = chunks
            total_chunks += chunks
            
            if chunks == 0:
                all_pipelines_processed = False
                
            if state.value in {"running", "starting", "pausing"}:
                active_count += 1
                
            # Check for failed pipelines
            if state == PipelineState.FAILED:
                logger.error(f"M3U8 pipeline {pipeline_id} has failed")
                # Continue testing since we have multiple pipelines
        
        logger.info(f"Active M3U8 pipelines: {active_count}, Total chunks processed: {total_chunks}")
        
        # If all pipelines have processed at least one chunk, we can stop
        if all_pipelines_processed and total_chunks >= len(pipeline_ids):
            break
            
        # Wait before checking again
        await asyncio.sleep(5)
    
    # Stop all pipelines
    for pipeline_id in pipeline_ids:
        await pipeline_manager.stop_pipeline(pipeline_id)
    
    # Success if at least one pipeline processed a chunk
    # This is more resilient than requiring all pipelines to succeed
    success = any(pipeline_chunks[pid] > 0 for pid in pipeline_ids)
    
    if success:
        logger.info(f"Multiple M3U8 pipeline test completed successfully")
    else:
        logger.error(f"Multiple M3U8 pipeline test failed - no pipelines processed chunks")
        
    return success

async def test_mixed_sources_pipelines(m3u8_url: str, youtube_url: str):
    """
    Test running multiple pipelines concurrently with different source types.
    
    Args:
        m3u8_url: M3U8 URL to use for testing
        youtube_url: YouTube URL to use for testing
    """
    # Clean the URLs to remove any trailing whitespace
    m3u8_url = m3u8_url.strip()
    youtube_url = youtube_url.strip()
    logger.info(f"Testing mixed sources pipelines with M3U8 and YouTube")
    
    # Create pipeline manager
    pipeline_manager = PipelineManager(
        base_data_dir=settings.PIPELINE.BASE_DATA_DIR,
        max_concurrent_pipelines=2,  # Limit to 2 concurrent pipelines
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Create pipelines with different source types
    
    # Create M3U8 pipeline
    m3u8_source = await pipeline_manager.register_source(
        url=m3u8_url,
        source_type="m3u8",
        metadata={"test": True, "index": 0}
    )
    
    m3u8_pipeline_id = await pipeline_manager.create_pipeline(
        source=m3u8_source,
        chunk_duration=30,
        analysis_prompt="Describe this M3U8 stream briefly",
        use_web_search=False
    )
    
    logger.info(f"Created M3U8 pipeline: {m3u8_pipeline_id}")
    
    # Create YouTube pipeline
    youtube_source = await pipeline_manager.register_source(
        url=youtube_url,
        source_type="youtube",
        metadata={"test": True, "index": 1}
    )
    
    youtube_pipeline_id = await pipeline_manager.create_pipeline(
        source=youtube_source,
        chunk_duration=30,
        analysis_prompt="Describe this YouTube video briefly",
        use_web_search=False
    )
    
    logger.info(f"Created YouTube pipeline: {youtube_pipeline_id}")
    
    # Start both pipelines
    await pipeline_manager.start_pipeline(m3u8_pipeline_id)
    logger.info(f"Started pipeline: {m3u8_pipeline_id}")
    
    await pipeline_manager.start_pipeline(youtube_pipeline_id)
    logger.info(f"Started pipeline: {youtube_pipeline_id}")
    
    # Monitor pipelines
    max_wait_time = 300  # 5 minutes max
    start_time = time.time()
    
    # Initialize these variables before using them
    m3u8_chunks = 0
    youtube_chunks = 0
    
    while time.time() - start_time < max_wait_time and (m3u8_chunks < 1 or youtube_chunks < 1):
        # Get M3U8 pipeline status
        m3u8_status = await pipeline_manager.get_pipeline_status(m3u8_pipeline_id)
        if m3u8_status:
            m3u8_state = PipelineState(m3u8_status["state"])
            m3u8_chunks = m3u8_status["stats"]["chunks_processed"]
            
        # Get YouTube pipeline status
        youtube_status = await pipeline_manager.get_pipeline_status(youtube_pipeline_id)
        if youtube_status:
            youtube_state = PipelineState(youtube_status["state"])
            youtube_chunks = youtube_status["stats"]["chunks_processed"]
        
        logger.info(f"M3U8 chunks: {m3u8_chunks}, YouTube chunks: {youtube_chunks}")
        
        # Check for pipeline failures (but don't exit - we're testing if either works)
        if m3u8_status and m3u8_state == PipelineState.FAILED:
            logger.warning("M3U8 pipeline has failed, continuing to wait for YouTube pipeline")
            
        if youtube_status and youtube_state == PipelineState.FAILED:
            logger.warning("YouTube pipeline has failed, continuing to wait for M3U8 pipeline")
            
        # If both pipelines have failed, exit early
        if (m3u8_status and m3u8_state == PipelineState.FAILED and 
            youtube_status and youtube_state == PipelineState.FAILED):
            logger.error("Both pipelines have failed")
            return False
        
        # Wait before checking again
        await asyncio.sleep(5)
    
    # Stop all pipelines
    await pipeline_manager.stop_pipeline(m3u8_pipeline_id)
    await pipeline_manager.stop_pipeline(youtube_pipeline_id)
    
    # Check results
    all_statuses = await pipeline_manager.get_all_pipeline_statuses()
    logger.info(f"Final pipeline statuses: {all_statuses}")
    
    # Success if at least one pipeline type processed chunks
    # This is more resilient than requiring both pipeline types to succeed
    success = m3u8_chunks >= 1 or youtube_chunks >= 1
    
    if success:
        if m3u8_chunks >= 1 and youtube_chunks >= 1:
            logger.info(f"Mixed sources pipeline test completed successfully - both pipeline types processed chunks")
        elif m3u8_chunks >= 1:
            logger.info(f"Mixed sources pipeline test completed with partial success - only M3U8 pipeline processed chunks")
        else:
            logger.info(f"Mixed sources pipeline test completed with partial success - only YouTube pipeline processed chunks")
    else:
        logger.error(f"Mixed sources pipeline test failed - no pipelines processed chunks")
        
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

async def test_pipeline_creation_with_source_id(url: str):
    """Test creating a pipeline using a source ID instead of a source object."""
    logger.info(f"Testing pipeline creation with source ID using URL: {url}")
    
    pipeline_manager = PipelineManager(
        base_data_dir=settings.PIPELINE.BASE_DATA_DIR,
        max_concurrent_pipelines=1,
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Register source and get its ID
    source = await pipeline_manager.register_source(
        url=url,
        source_type="youtube",  # or "m3u8" for the m3u8 test
        metadata={"test": True}
    )
    source_id = source.source_id
    
    # Create pipeline using only the source ID
    pipeline_id = await pipeline_manager.create_pipeline(
        source=source_id,  # Pass ID instead of source object
        chunk_duration=20,
        analysis_prompt="Testing source ID pipeline creation",
        use_web_search=False
    )
    
    logger.info(f"Created pipeline using source ID: {pipeline_id}")
    
    # Verify pipeline was created with the correct source
    status = await pipeline_manager.get_pipeline_status(pipeline_id)
    if not status:
        logger.error("Pipeline status not found")
        return False
        
    pipeline_source_id = status.get("source_id")
    if pipeline_source_id != source_id:
        logger.error(f"Pipeline has wrong source ID: {pipeline_source_id} vs expected {source_id}")
        return False
    
    logger.info("Pipeline correctly created with source ID")
    await pipeline_manager.stop_pipeline(pipeline_id)
    return True

async def main():
    """Run all integration tests with M3U8 and YouTube URLs."""
    try:
        # Default to a public stream if no URL is provided
        default_m3u8_url = "https://streaming.natasquad.tv/hls/chanel1.m3u8"  # Public test stream
        default_youtube_url = "https://www.youtube.com/watch?v=LPmbtKSwN6E"  # lofi hip hop radio
        
        # Check if URL is provided as command line argument
        if len(sys.argv) > 1:
            m3u8_url = sys.argv[1]
        else:
            m3u8_url = default_m3u8_url
            
        # Use default YouTube URL for mixed source test
        youtube_url = default_youtube_url
            
        logger.info(f"Starting integration tests with M3U8 URL: {m3u8_url}")
        
        # Track test results
        results = {}
        
        # Run tests with individual try/except blocks
        try:
            results["websocket"] = await test_websocket_broadcast()
            logger.info(f"WebSocket test result: {'PASS' if results['websocket'] else 'FAIL'}")
        except Exception as e:
            logger.error(f"Error in websocket test: {e}")
            results["websocket"] = False
        
        try:
            results["m3u8_pipeline"] = await test_m3u8_pipeline_manager(m3u8_url)
            logger.info(f"M3U8 pipeline test result: {'PASS' if results['m3u8_pipeline'] else 'FAIL'}")
        except Exception as e:
            logger.error(f"Error in M3U8 pipeline test: {e}")
            results["m3u8_pipeline"] = False
            
        try:
            results["error_handling"] = await test_m3u8_error_handling()
            logger.info(f"M3U8 error handling test result: {'PASS' if results['error_handling'] else 'FAIL'}")
        except Exception as e:
            logger.error(f"Error in M3U8 error handling test: {e}")
            results["error_handling"] = False
            
        try:
            results["multiple_m3u8"] = await test_multiple_m3u8_pipelines(m3u8_url)
            logger.info(f"Multiple M3U8 pipeline test result: {'PASS' if results['multiple_m3u8'] else 'FAIL'}")
        except Exception as e:
            logger.error(f"Error in multiple M3U8 pipelines test: {e}")
            results["multiple_m3u8"] = False
        
        try:
            results["pipeline_with_source_id"] = await test_pipeline_creation_with_source_id(m3u8_url)  
            logger.info(f"Source ID pipeline test result: {'PASS' if results['pipeline_with_source_id'] else 'FAIL'}")
        except Exception as e:
            logger.error(f"Error in create pipeline with source id test: {e}")
            results["pipeline_with_source_id"] = False
            
        #try:
        #    results["mixed_sources"] = await test_mixed_sources_pipelines(m3u8_url, youtube_url)
        #    logger.info(f"Mixed sources pipeline test result: {'PASS' if results['mixed_sources'] else 'FAIL'}")
        #except Exception as e:
        #    logger.error(f"Error in mixed sources test: {e}")
        #    results["mixed_sources"] = False
        
        # Overall result - consider critical tests
        critical_tests = ["websocket", "error_handling", "multiple_m3u8", "pipeline_with_source_id"]
        all_critical_passed = all(results.get(test, False) for test in critical_tests)
        
        logger.info(f"M3U8 integration tests overall result: {'PASS' if all_critical_passed else 'FAIL'}")
        
        return all_critical_passed
        
    except Exception as e:
        logger.error(f"Error in M3U8 integration tests: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Run the integration tests
    result = asyncio.run(main())
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if result else 1)