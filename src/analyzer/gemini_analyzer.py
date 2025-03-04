# src/analyzer/gemini_analyzer.py
import json
import logging
import time
import os
import hashlib
from datetime import datetime
from google import genai
from google.genai import types
from google.genai.errors import ServerError  # Only import ServerError
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AnalysisCache:
    """Simple cache for analysis results to avoid re-analyzing similar content."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[str, float]] = {}  # hash -> (result, timestamp)
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[str]:
        """Get an item from the cache if it exists and is not expired."""
        if key not in self.cache:
            self.misses += 1
            return None
            
        result, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl_seconds:
            # Expired
            del self.cache[key]
            self.misses += 1
            return None
            
        self.hits += 1
        return result
        
    def set(self, key: str, value: str) -> None:
        """Store an item in the cache, evicting old entries if necessary."""
        if len(self.cache) >= self.max_size:
            # Evict oldest entry
            oldest_key = None
            oldest_time = float('inf')
            for k, (_, timestamp) in self.cache.items():
                if timestamp < oldest_time:
                    oldest_time = timestamp
                    oldest_key = k
                    
            if oldest_key:
                del self.cache[oldest_key]
                
        self.cache[key] = (value, time.time())
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }

class GeminiVideoAnalyzer:
    """An enhanced class to analyze video content using Gemini Vision."""
    
    def __init__(self, 
                api_key: str, 
                max_retries: int = 3, 
                retry_delay: int = 15,
                enable_caching: bool = True,
                cache_size: int = 100,
                base_data_dir: str = "data",
                export_responses: bool = False):
        """
        Initialize the GeminiVideoAnalyzer.
        
        Args:
            api_key (str): Google Gemini API key
            max_retries (int): Maximum number of retries for failed operations
            retry_delay (int): Delay in seconds between retries
            enable_caching (bool): Whether to cache results to avoid re-analyzing similar content
            cache_size (int): Maximum number of entries in the cache
            base_data_dir (str): Base directory for storing responses
            export_responses (bool): Whether to export full responses to files
        """
        self.client = genai.Client(api_key=api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_caching = enable_caching
        self.cache = AnalysisCache(max_size=cache_size) if enable_caching else None
        self.base_data_dir = base_data_dir
        self.export_responses = export_responses
        self._setup_logging()
        
        # Performance tracking
        self.success_count = 0
        self.failure_count = 0
        self.total_processing_time = 0
        self.average_processing_time = 0
        
    def _setup_logging(self) -> None:
        """Configure logging for the class."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _get_video_fingerprint(self, video_path: str) -> str:
        """
        Generate a fingerprint for a video file to use as a cache key.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            str: A hash representing the video content
        """
        # Get file stats to create a simple fingerprint (size + mtime)
        try:
            stat = os.stat(video_path)
            # Using file size and modification time as a simple fingerprint
            content = f"{os.path.basename(video_path)}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not generate fingerprint for {video_path}: {e}")
            # Fallback to just the filename if we can't get stats
            return hashlib.md5(os.path.basename(video_path).encode()).hexdigest()

    def _validate_video_file(self, video_path: str, retry_count: int = 3, retry_delay: int = 2) -> None:
        """
        Validate that the video file exists and is accessible.
        
        Args:
            video_path (str): Path to the video file
            retry_count (int): Number of retries for small files
            retry_delay (int): Delay between retries in seconds
            
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
        
        # Skip file size check in test mode
        if os.getenv("GEMINI_TEST_MODE", "").lower() == "true":
            return
            
        # Check file size with retries
        min_size = int(os.getenv("MIN_VIDEO_SIZE_BYTES", "1024"))  # Default to 1KB for production
        
        for attempt in range(retry_count):
            file_size = os.path.getsize(video_path)
            
            if file_size >= min_size:
                return  # File size is good
                
            if attempt < retry_count - 1:
                self.logger.warning(
                    f"Video file too small ({file_size} bytes), waiting for it to grow. "
                    f"Attempt {attempt+1}/{retry_count}"
                )
                time.sleep(retry_delay)
            else:
                # Last attempt failed
                raise ValueError(f"Video file is too small after {retry_count} retries: {file_size} bytes")

    def _backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.
        
        Args:
            attempt (int): The current attempt number (0-based)
            
        Returns:
            float: Delay in seconds
        """
        # Exponential backoff with jitter
        import random
        base_delay = min(self.retry_delay * (2 ** attempt), 60)  # Cap at 60 seconds
        jitter = random.uniform(0, 0.1 * base_delay)  # 10% jitter
        return base_delay + jitter

    def _is_server_error(self, exception: Exception) -> bool:
        """
        Check if an exception is a server-side error that should be retried.
        
        Args:
            exception: The exception to check
            
        Returns:
            bool: True if it's a server error, False otherwise
        """
        # Check if it's a ServerError from the Google API
        if isinstance(exception, ServerError):
            return True
            
        # Check for common error messages in other exception types
        error_msg = str(exception).lower()
        server_error_indicators = [
            "server error", 
            "internal server error",
            "rate limit",
            "too many requests",
            "timeout", 
            "connection",
            "network",
            "503",
            "500",
            "429"
        ]
        
        return any(indicator in error_msg for indicator in server_error_indicators)

    # Enhanced methods for GeminiVideoAnalyzer class with detailed response logging

    def analyze_video(self, video_path: str, prompt: str) -> str:
        """
        Analyze a video file using Gemini Vision.
        
        Args:
            video_path (str): Path to the video file
            prompt (str): Prompt for analysis
            
        Returns:
            str: Analysis result from Gemini
        """
        self.logger.info(f"Starting analysis of {video_path}")
        start_time = time.time()
        
        try:
            # Validate video file
            self._validate_video_file(video_path)
            
            # Check cache if enabled
            if self.enable_caching:
                cache_key = self._get_video_fingerprint(video_path)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.logger.info(f"Cache hit for {video_path}")
                    return cached_result
            
            # For testing/development, return a dummy result to avoid API calls
            if os.getenv("GEMINI_TEST_MODE", "").lower() == "true":
                self.logger.info("Test mode active, returning dummy result")
                result = f"Test analysis of {os.path.basename(video_path)} - Prompt: {prompt[:50]}..."
                return result
            
            # Upload video to Gemini with retries
            video_file = None
            upload_success = False
            
            for attempt in range(self.max_retries):
                try:
                    self.logger.info(f"Uploading video: {video_path} (attempt {attempt + 1}/{self.max_retries})")
                    video_file = self.client.files.upload(file=video_path)
                    upload_success = True
                    self.logger.info(f"Upload successful. File ID: {video_file.name}")
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        self.logger.error(f"Failed to upload video after {self.max_retries} attempts: {e}")
                        raise
                    
                    if self._is_server_error(e):
                        delay = self._backoff_delay(attempt)
                        self.logger.warning(f"Upload attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        # Not a server error, probably a client issue - don't retry
                        self.logger.error(f"Client error uploading video: {e}")
                        raise
            
            if not upload_success or not video_file:
                raise Exception("Failed to upload video after all retries")
            
            # Wait for processing
            self._wait_for_processing(video_file)
            
            # Generate content with Gemini
            self.logger.info("Generating analysis...")
            response = None
            
            for attempt in range(self.max_retries):
                try:
                    self.logger.info(f"Sending prompt to Gemini: {prompt[:100]}...")
                    response = self.client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[video_file, prompt]
                    )
                    self.logger.info("Received response from Gemini")
                    # Log detailed response information
                    self._log_gemini_response(response, video_path)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        self.logger.error(f"Failed to generate content after {self.max_retries} attempts: {e}")
                        raise
                    
                    if self._is_server_error(e):
                        delay = self._backoff_delay(attempt)
                        self.logger.warning(f"Generation attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        # Not a server error, probably a client issue - don't retry
                        self.logger.error(f"Client error generating content: {e}")
                        raise
            
            if not response:
                raise Exception("Failed to generate content after all retries")
                
            result = response.text
            
            # Cache the result if enabled
            if self.enable_caching:
                cache_key = self._get_video_fingerprint(video_path)
                self.cache.set(cache_key, result)
            
            # Update performance stats
            self.success_count += 1
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.average_processing_time = self.total_processing_time / (self.success_count + self.failure_count)
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.average_processing_time = self.total_processing_time / (self.success_count + self.failure_count)
            
            self.logger.error(f"Error analyzing video {video_path}: {e}")
            raise
        finally:
            # Clean up the video file if it was uploaded
            if 'video_file' in locals() and video_file:
                try:
                    self.client.files.delete(name=video_file.name)
                except:
                    self.logger.warning("Failed to clean up video file from Gemini storage")
            
            self.logger.info(f"Analysis of {video_path} completed in {time.time() - start_time:.2f}s")

    def analyze_video_with_web_search(self, video_path: str, prompt: str) -> str:
        """
        Analyze a video file using Gemini Vision with web search capability.
        
        Args:
            video_path (str): Path to the video file
            prompt (str): Prompt for analysis
            
        Returns:
            str: Analysis result with web-enriched information
        """
        self.logger.info(f"Starting web-search analysis of {video_path}")
        start_time = time.time()
        
        try:
            # Validate video file
            self._validate_video_file(video_path)
            
            # Check cache if enabled
            if self.enable_caching:
                # Add a "web" prefix to distinguish from non-web-search analyses
                cache_key = "web_" + self._get_video_fingerprint(video_path)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.logger.info(f"Cache hit for web-search analysis of {video_path}")
                    return cached_result
            
            # For testing/development, return a dummy result to avoid API calls
            if os.getenv("GEMINI_TEST_MODE", "").lower() == "true":
                self.logger.info("Test mode active, returning dummy result with web search")
                result = f"Test web-search analysis of {os.path.basename(video_path)} - Prompt: {prompt[:50]}..."
                return result
            
            # Upload video with retries
            video_file = None
            upload_success = False
            
            for attempt in range(self.max_retries):
                try:
                    self.logger.info(f"Uploading video for web search: {video_path} (attempt {attempt + 1}/{self.max_retries})")
                    video_file = self.client.files.upload(file=video_path)
                    upload_success = True
                    self.logger.info(f"Upload successful for web search. File ID: {video_file.name}")
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        self.logger.error(f"Failed to upload video after {self.max_retries} attempts: {e}")
                        raise
                    
                    if self._is_server_error(e):
                        delay = self._backoff_delay(attempt)
                        self.logger.warning(f"Upload attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        # Not a server error, probably a client issue - don't retry
                        self.logger.error(f"Client error uploading video: {e}")
                        raise
            
            if not upload_success or not video_file:
                raise Exception("Failed to upload video after all retries")
            
            # Wait for processing
            self._wait_for_processing(video_file)
            
            # Configure search tool
            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
            
            # Generate content with web search
            self.logger.info("Generating analysis with web search...")
            response = None
            
            for attempt in range(self.max_retries):
                try:
                    self.logger.info(f"Sending prompt to Gemini with web search: {prompt[:100]}...")
                    response = self.client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[video_file, prompt],
                        config=config
                    )
                    self.logger.info("Received response from Gemini with web search")
                    # Log detailed response information
                    self._log_gemini_response(response, video_path, web_search=True)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        self.logger.error(f"Failed to generate content with web search after {self.max_retries} attempts: {e}")
                        raise
                    
                    if self._is_server_error(e):
                        delay = self._backoff_delay(attempt)
                        self.logger.warning(f"Generation attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        # Not a server error, probably a client issue - don't retry
                        self.logger.error(f"Client error generating content: {e}")
                        raise
            
            if not response:
                raise Exception("Failed to generate content with web search after all retries")
                
            result = response.text
            
            # Cache the result if enabled
            if self.enable_caching:
                cache_key = "web_" + self._get_video_fingerprint(video_path)
                self.cache.set(cache_key, result)
            
            # Update performance stats
            self.success_count += 1
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.average_processing_time = self.total_processing_time / (self.success_count + self.failure_count)
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.average_processing_time = self.total_processing_time / (self.success_count + self.failure_count)
            
            self.logger.error(f"Error analyzing video with web search {video_path}: {e}")
            raise
        finally:
            # Clean up the video file if it was uploaded
            if 'video_file' in locals() and video_file:
                try:
                    self.client.files.delete(name=video_file.name)
                except:
                    self.logger.warning("Failed to clean up video file from Gemini storage")
            
            self.logger.info(f"Web-search analysis of {video_path} completed in {time.time() - start_time:.2f}s")

    def _log_gemini_response(self, response, video_path, web_search=False):
        """
        Log detailed information about the Gemini response.
        
        Args:
            response: The response object from Gemini
            video_path: Path to the video file that was analyzed
            web_search: Whether web search was used
        """
        try:
            # Create a readable filename for logging
            video_name = os.path.basename(video_path)
            
            # Log basic info
            search_type = "with web search" if web_search else "standard"
            self.logger.info(f"Gemini {search_type} response for {video_name}:")
            
            # Log the full response text
            self.logger.info(f"Response text:\n{response.text[:500]}...")
            
            # Try to extract and log additional metadata if available
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                self.logger.info(f"Content rating: {getattr(candidate, 'content_trust', 'N/A')}")
                self.logger.info(f"Response length: {len(response.text)} characters")
                
                # Log any safety ratings if present
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    self.logger.info("Safety ratings:")
                    for rating in candidate.safety_ratings:
                        self.logger.info(f"- {rating.category}: {rating.probability}")
                
                # Log any citation metadata if present
                if hasattr(candidate, 'citation_metadata') and candidate.citation_metadata:
                    self.logger.info("Citations found in response")
            
            # Log any web search information if available
            if web_search and hasattr(response, 'tools_info'):
                self.logger.info("Web search information:")
                self.logger.info(f"Search details: {getattr(response, 'tools_info', 'No search details available')}")
                
        except Exception as e:
            self.logger.warning(f"Error logging Gemini response details: {e}")
            
    def _export_gemini_response(self, response, video_path, web_search=False):
        """
        Export the full Gemini response to a file for detailed analysis.
        
        Args:
            response: The response object from Gemini
            video_path: Path to the video file that was analyzed
            web_search: Whether web search was used
        """
        try:
            # Create a unique filename based on the video and timestamp
            video_name = os.path.basename(video_path).replace('.mp4', '')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            search_type = "websearch" if web_search else "standard"
            
            # Create responses directory if it doesn't exist
            responses_dir = os.path.join(self.base_data_dir, "gemini_responses")
            os.makedirs(responses_dir, exist_ok=True)
            
            # Create the response file
            filename = f"{responses_dir}/{video_name}_{search_type}_{timestamp}.json"
            
            # Extract all relevant info from the response object
            response_data = {
                "timestamp": timestamp,
                "video_path": video_path,
                "analysis_type": "web_search" if web_search else "standard",
                "text": response.text,
                "metadata": {}
            }
            
            # Try to add additional metadata if available
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Add safety ratings if present
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    response_data["metadata"]["safety_ratings"] = [
                        {"category": rating.category, "probability": rating.probability}
                        for rating in candidate.safety_ratings
                    ]
                
                # Add citation metadata if present
                if hasattr(candidate, 'citation_metadata') and candidate.citation_metadata:
                    response_data["metadata"]["citations"] = str(candidate.citation_metadata)
                    
            # Add tools info for web search
            if web_search and hasattr(response, 'tools_info'):
                response_data["metadata"]["tools_info"] = str(response.tools_info)
            
            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Exported Gemini response to {filename}")
            
        except Exception as e:
            self.logger.warning(f"Error exporting Gemini response to file: {e}")

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
                
            elapsed = time.time() - start_time
            self.logger.info(f"Waiting for video processing... ({elapsed:.1f}s elapsed)")
            time.sleep(10)
            
            try:
                video_file = self.client.files.get(name=video_file.name)
            except Exception as e:
                self.logger.warning(f"Error checking processing status: {e}")
                time.sleep(5)  # Brief delay before retry
                continue
            
        if video_file.state == "FAILED":
            raise ValueError("Video processing failed")
        elif video_file.state != "ACTIVE":
            raise ValueError(f"Unexpected video state: {video_file.state}")
            
        self.logger.info(f"Video processing completed in {time.time() - start_time:.1f}s")
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the analyzer."""
        stats = {
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_count": self.success_count + self.failure_count,
            "success_rate": self.success_count / (self.success_count + self.failure_count) if (self.success_count + self.failure_count) > 0 else 0,
            "average_processing_time": self.average_processing_time,
            "total_processing_time": self.total_processing_time
        }
        
        if self.enable_caching:
            stats["cache"] = self.cache.get_stats()
            
        return stats