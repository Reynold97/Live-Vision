# src/recorder/base_recorder.py
from abc import ABC, abstractmethod
import logging
import os
import time
from typing import Dict, Any, Optional

class ChunkingProgress:
    """Tracks the progress of a chunking operation."""
    
    def __init__(self):
        self.start_time = time.time()
        self.chunks_created = 0
        self.last_chunk_time = time.time()
        self.errors = 0
        self.status = "initialized"  # initialized, running, completed, failed
        self.last_error = None
        self.is_live = False
        self.chunk_timestamps = {}  # chunk filename -> creation timestamp
        
    def update(self, new_chunk: bool = False, chunk_name: Optional[str] = None, 
               error: Optional[Exception] = None, is_live: Optional[bool] = None) -> None:
        """Update progress information."""
        if new_chunk and chunk_name:
            self.chunks_created += 1
            self.last_chunk_time = time.time()
            self.chunk_timestamps[chunk_name] = time.time()
            
        if error:
            self.errors += 1
            self.last_error = str(error)
            
        if is_live is not None:
            self.is_live = is_live
            
    def get_status(self) -> Dict[str, Any]:
        """Get current status as a dictionary."""
        return {
            "start_time": self.start_time,
            "elapsed_time": time.time() - self.start_time,
            "chunks_created": self.chunks_created,
            "last_chunk_time": self.last_chunk_time,
            "time_since_last_chunk": time.time() - self.last_chunk_time,
            "errors": self.errors,
            "last_error": self.last_error,
            "status": self.status,
            "is_live": self.is_live,
            "chunk_count": len(self.chunk_timestamps)
        }
        
    def get_chunk_age(self, chunk_name: str) -> Optional[float]:
        """Get the age of a chunk in seconds."""
        if chunk_name in self.chunk_timestamps:
            return time.time() - self.chunk_timestamps[chunk_name]
        return None

class BaseStreamRecorder(ABC):
    """Abstract base class for all stream recorders."""
    
    def __init__(self, base_data_folder="data"):
        """
        Initialize the base recorder.
        
        Args:
            base_data_folder (str): Base folder where all video chunks will be stored
        """
        self.base_data_folder = base_data_folder
        self.progress_trackers = {}
        self.stop_signals = {}
        self.active_processes = {}
        
        # Set up logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for the class."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def process_video(self, url: str, chunk_duration: int, output_dir: Optional[str] = None) -> str:
        """
        Process a video stream into chunks.
        
        Args:
            url: Stream URL to process
            chunk_duration: Duration of each chunk in seconds
            output_dir: Optional custom output directory
            
        Returns:
            str: Path to the output directory
        """
        pass
    
    @abstractmethod
    def stop_processing(self, url: str) -> bool:
        """
        Stop an active processing job.
        
        Args:
            url: URL of the stream to stop processing
            
        Returns:
            bool: True if successfully stopped, False otherwise
        """
        pass
    
    @abstractmethod
    def _test_url_accessibility(self, url: str, url_hash: str) -> bool:
        """
        Test if a URL is accessible.
        
        Args:
            url: URL to test
            url_hash: Hash of the URL for tracking
            
        Returns:
            bool: True if accessible, False otherwise
        """
        pass
        
    def get_progress(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get progress information for a URL.
        
        Args:
            url: URL to get progress for
            
        Returns:
            Dict or None: Progress information or None if not found
        """
        url_hash = self._get_url_hash(url)
        if url_hash in self.progress_trackers:
            return self.progress_trackers[url_hash].get_status()
        return None
    
    def _create_output_directory(self, custom_dir: Optional[str] = None) -> str:
        """
        Create a nested directory structure for video chunks.
        Format: base_data_folder/YYYY_MM_DD/HH_MM/
        
        Args:
            custom_dir: Optional custom directory to use instead of generating one
            
        Returns:
            str: Path to the created directory
        """
        # Keep existing implementation
        # This functionality is common to all recorders
        
    def _get_url_hash(self, url: str) -> str:
        """Generate a hash for the URL to use as a unique identifier."""
        # Keep existing implementation
        # This functionality is common to all recorders