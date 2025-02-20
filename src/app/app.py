import streamlit as st
import asyncio
import os
from src.recorder.stream_recorder import YouTubeChunker
from src.analyzer.gemini_analyzer import GeminiVideoAnalyzer
from src.pipeline.video_pipeline import VideoPipeline

async def run_pipeline(url: str, chunk_duration: int, api_key: str):
    """Run the video analysis pipeline."""
    # Initialize components
    chunker = YouTubeChunker()
    analyzer = GeminiVideoAnalyzer(api_key=api_key)
    pipeline = VideoPipeline(chunker, analyzer)
    
    try:
        # Start pipeline
        await pipeline.start_pipeline(
            url=url,
            chunk_duration=chunk_duration,
            analysis_prompt="Describe what happens in this video segment, including any significant events, conversations, or actions.",
            use_web_search=True
        )
    except Exception as e:
        st.error(f"Error in pipeline: {e}")

def main():
    st.title("YouTube Video Analysis Pipeline")
    st.write("Enter a YouTube URL and chunk duration to analyze the video in real-time.")
    
    # API Key input
    api_key = st.text_input("Enter your Google API Key", type="password")
    if not api_key:
        st.warning("Please enter your Google API Key to continue")
        return
        
    # Form for input parameters
    with st.form("video_params"):
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        chunk_duration = st.slider("Chunk Duration (seconds)", min_value=30, max_value=300, value=30, step=5)
        started = st.form_submit_button("Start Analysis")
    
    if started and url and api_key:
        # Create a placeholder for analysis results
        results_container = st.container()
        stop_button = st.button("Stop Analysis")
        
        with results_container:
            st.write("### Analysis Results")
            
            # Create columns for timestamps and analysis
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.write("**Time**")
            with col2:
                st.write("**Analysis**")
                
            # Run the pipeline
            try:
                asyncio.run(run_pipeline(url, chunk_duration, api_key))
            except KeyboardInterrupt:
                st.info("Analysis stopped by user")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()