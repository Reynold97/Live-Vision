import logging
import time
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class GeminiVideoAnalyzer:
    """A class to analyze video content using Gemini Vision."""
    
    def __init__(self, api_key: str, max_retries: int = 3, retry_delay: int = 15):
        """
        Initialize the GeminiVideoAnalyzer.
        
        Args:
            api_key (str): Google Gemini API key
            max_retries (int): Maximum number of retries for failed operations
            retry_delay (int): Delay in seconds between retries
        """
        self.client = genai.Client(api_key=api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure logging for the class."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _validate_video_file(self, video_path: str) -> None:
        """
        Validate that the video file exists and is accessible.
        
        Args:
            video_path (str): Path to the video file
            
        Raises:
            FileNotFoundError: If the video file doesn't exist
            ValueError: If the file is not a video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Check if file has a video extension
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        file_ext = os.path.splitext(video_path)[1].lower()
        if file_ext not in valid_extensions:
            raise ValueError(f"Invalid video format. Supported formats: {valid_extensions}")

    def analyze_video(self, video_path: str, prompt: str) -> str:
        """
        Analyze a video file using Gemini Vision.
        Note: This method is designed to be run in a separate thread via asyncio.to_thread
        
        Args:
            video_path (str): Path to the video file
            prompt (str): Prompt for analysis
            
        Returns:
            str: Analysis result from Gemini
        """
        try:
            # Validate video file
            self._validate_video_file(video_path)
            
            # Upload video to Gemini with retries
            video_file = None
            for attempt in range(self.max_retries):
                try:
                    video_file = self.client.files.upload(file=video_path)
                    break
                except ServerError as e:
                    if attempt == self.max_retries - 1:
                        raise
                    self.logger.warning(f"Upload attempt {attempt + 1} failed, retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
            
            if not video_file:
                raise Exception("Failed to upload video after all retries")
            
            # Wait for processing
            self._wait_for_processing(video_file)
            
            # Generate content with Gemini
            self.logger.info("Generating analysis...")
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[video_file, prompt]
            )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error analyzing video: {e}")
            raise
        finally:
            # Clean up the video file if it was uploaded
            if video_file:
                try:
                    self.client.files.delete(name=video_file.name)
                except:
                    self.logger.warning("Failed to clean up video file from Gemini storage")

    def analyze_video_with_web_search(self, video_path: str, prompt: str) -> str:
        """
        Analyze a video file using Gemini Vision with web search capability.
        Note: This method is designed to be run in a separate thread via asyncio.to_thread
        
        Args:
            video_path (str): Path to the video file
            prompt (str): Prompt for analysis
            
        Returns:
            str: Analysis result with web-enriched information
        """
        try:
            # Validate video file
            self._validate_video_file(video_path)
            
            # Upload video with retries
            video_file = None
            for attempt in range(self.max_retries):
                try:
                    video_file = self.client.files.upload(file=video_path)
                    break
                except ServerError as e:
                    if attempt == self.max_retries - 1:
                        raise
                    self.logger.warning(f"Upload attempt {attempt + 1} failed, retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
            
            if not video_file:
                raise Exception("Failed to upload video after all retries")
            
            # Wait for processing
            self._wait_for_processing(video_file)
            
            # Configure search tool
            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search={})]
            )
            
            # Generate content with web search
            self.logger.info("Generating analysis with web search...")
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[video_file, prompt],
                config=config
            )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error analyzing video with web search: {e}")
            raise
        finally:
            # Clean up the video file if it was uploaded
            if video_file:
                try:
                    self.client.files.delete(name=video_file.name)
                except:
                    self.logger.warning("Failed to clean up video file from Gemini storage")

    def _wait_for_processing(self, video_file: Any) -> None:
        """
        Wait for video processing to complete with timeout.
        
        Args:
            video_file: The video file object from Gemini upload
            
        Raises:
            ValueError: If video processing fails
            TimeoutError: If processing takes too long
        """
        start_time = time.time()
        timeout = 300  # 5 minutes timeout
        
        while video_file.state == "PROCESSING":
            if time.time() - start_time > timeout:
                raise TimeoutError("Video processing timed out")
                
            self.logger.info("Waiting for video processing...")
            time.sleep(10)
            
            try:
                video_file = self.client.files.get(name=video_file.name)
            except ServerError as e:
                self.logger.warning(f"Error checking processing status: {e}")
                time.sleep(5)  # Brief delay before retry
                continue
            
        if video_file.state == "FAILED":
            raise ValueError("Video processing failed")
        elif video_file.state != "ACTIVE":
            raise ValueError(f"Unexpected video state: {video_file.state}")

# Example usage:
if __name__ == "__main__":
    import os
    
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    # Initialize the analyzer
    analyzer = GeminiVideoAnalyzer(
        api_key=api_key,
        max_retries=3,
        retry_delay=15
    )
    
    try:
        # Basic video analysis
        result = analyzer.analyze_video(
            video_path=r"data\2025_02_20\16_48\chunk_000.mp4",
            prompt="Describe what happens in this video"
        )
        print("Basic Analysis Result:", result)
        
        result2 = analyzer.analyze_video_with_web_search(
            video_path=r"data\2025_02_20\16_48\chunk_000.mp4",
            prompt="Describe what happens in this video and offer web relevant context including URL."
        )
        print("Basic Analysis Result:", result2)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except ServerError as e:
        print(f"Gemini API Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")