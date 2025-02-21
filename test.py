import os
import asyncio
from src.recorder.stream_recorder import YouTubeChunker
from src.analyzer.gemini_analyzer import GeminiVideoAnalyzer
from src.pipeline.video_pipeline import VideoPipeline
from dotenv import load_dotenv

load_dotenv()

async def main():
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Initialize components
    chunker = YouTubeChunker()
    analyzer = GeminiVideoAnalyzer(api_key)
    
    # Create pipeline
    pipeline = VideoPipeline(chunker, analyzer)
    
    try:
        # Start pipeline with web search enabled
        await pipeline.start_pipeline(
            url="https://www.youtube.com/watch?v=JKvvJttjslg",
            chunk_duration=30,
            analysis_prompt="Describe the events in this video segment",
            use_web_search=True  # Enable web search for analysis
        )
    except KeyboardInterrupt:
        await pipeline.stop_pipeline()

# Run the pipeline
asyncio.run(main())