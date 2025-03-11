from .base_recorder import BaseStreamRecorder, ChunkingProgress
import subprocess
import os
import shutil
import signal
import platform
import hashlib
from datetime import datetime
import logging
import time
from typing import Optional, Dict, Any, List, Callable
import asyncio
import threading

class YouTubeChunker(BaseStreamRecorder):
    """An enhanced class to download YouTube videos and split them into chunks."""
    
    def __init__(self, base_data_folder="data", cookies_path=None):
        """
        Initialize the YouTubeChunker.
        
        Args:
            base_data_folder (str): Base folder where all video chunks will be stored
            cookies_path (str): Path to the cookies file for YouTube authentication
        """
        self.base_data_folder = base_data_folder
        self.cookies_path = cookies_path or os.path.join(os.path.expanduser("~"), "cookies", "youtube.txt")
        
        # Set up logging first
        self._setup_logging()
        
        # Check dependencies
        self._check_dependencies()
        
        # Initialize properties
        self.active_processes: Dict[str, subprocess.Popen] = {}  # yt-dlp processes
        self.ffmpeg_processes: Dict[str, subprocess.Popen] = {}  # Track ffmpeg processes
        self.progress_trackers: Dict[str, ChunkingProgress] = {}
        self.stop_signals: Dict[str, bool] = {}
        
        # Check if we're in development mode
        self.dev_mode = os.getenv("ENVIRONMENT", "").lower() == "development"
        
        # Check cookie configuration but don't let it fail initialization
        try:
            self.cookie_info = self.check_cookie_configuration()
        except Exception as e:
            self.logger.warning(f"Non-critical error checking cookie configuration: {e}")
            self.cookie_info = {"file_exists": False, "error": str(e)}
        
    def _setup_logging(self):
        """Configure logging for the class."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _check_dependencies(self):
        """Check if required external programs are installed."""
        try:
            subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except FileNotFoundError:
            raise SystemError(
                "Required dependencies not found. Please install yt-dlp and ffmpeg:\n"
                "pip install yt-dlp\n"
                "And install ffmpeg from your system package manager."
            )
        except subprocess.CalledProcessError:
            raise SystemError(
                "Error running dependencies. Please ensure yt-dlp and ffmpeg are properly installed."
            )
    
    def _create_output_directory(self, custom_dir: Optional[str] = None) -> str:
        """
        Create a nested directory structure for video chunks.
        Format: base_data_folder/YYYY_MM_DD/HH_MM/
        
        Args:
            custom_dir: Optional custom directory to use instead of generating one
            
        Returns:
            str: Path to the created directory
        """
        if custom_dir:
            output_dir = custom_dir
        else:
            now = datetime.now()
            date_folder = now.strftime('%Y_%m_%d')
            time_folder = now.strftime('%H_%M')
            
            output_dir = os.path.join(
                self.base_data_folder,
                date_folder,
                time_folder
            )
            
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Created directory structure: {output_dir}")
        return output_dir
    
    def _get_url_hash(self, url: str) -> str:
        """Generate a hash for the URL to use as a unique identifier."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def process_video(self, url: str, chunk_duration: int = 30, output_dir: Optional[str] = None) -> str:
        """
        Download a YouTube video and split it into chunks.
        
        Args:
            url: YouTube URL to process
            chunk_duration: Duration of each chunk in seconds
            output_dir: Optional custom output directory
            
        Returns:
            str: Path to the output directory
        """
        try:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"Invalid URL format: {url}")

            # Validate output directory
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = self._create_output_directory()
                
            self.logger.info(f"Created output directory: {output_dir}")
            
            # Initialize progress tracker
            url_hash = self._get_url_hash(url)
            self.progress_trackers[url_hash] = ChunkingProgress()
            self.stop_signals[url_hash] = False
            
            # Test yt-dlp can access the URL
            is_accessible = self._test_url_accessibility(url, url_hash)
            if not is_accessible:
                # The _test_url_accessibility function will raise an appropriate error
                raise ValueError(f"Could not access URL: {url}")
            
            # Start downloading and segmenting
            self.progress_trackers[url_hash].status = "running"
            self._download_and_segment(url, chunk_duration, output_dir, url_hash)
            
            # If we get here, process completed successfully
            self.progress_trackers[url_hash].status = "completed"
            self.logger.info("Video processing completed successfully")
            return output_dir
                
        except Exception as e:
            url_hash = self._get_url_hash(url)
            if url_hash in self.progress_trackers:
                self.progress_trackers[url_hash].update(error=e)
                self.progress_trackers[url_hash].status = "failed"
            self.logger.error(f"Error processing video: {e}")
            raise
    
    def _watch_for_new_chunks(self, output_dir: str, url_hash: str) -> None:
        """
        Watch for new chunks being created in the output directory.
        This is a synchronous version that runs in a separate thread.
        
        Args:
            output_dir: Directory to watch
            url_hash: Hash of the URL for tracking
        """
        known_chunks = set()
        
        while (
            url_hash in self.progress_trackers and 
            self.progress_trackers[url_hash].status == "running" and
            not self.stop_signals.get(url_hash, True)
        ):
            try:
                # Get the current chunks
                chunks = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
                
                # Check for new chunks
                for chunk in chunks:
                    if chunk not in known_chunks:
                        chunk_path = os.path.join(output_dir, chunk)
                        # Only count it if it's a valid file with some content
                        if os.path.isfile(chunk_path) and os.path.getsize(chunk_path) > 0:
                            known_chunks.add(chunk)
                            self.progress_trackers[url_hash].update(
                                new_chunk=True, 
                                chunk_name=chunk
                            )
                            self.logger.info(f"New chunk detected: {chunk}")
                    
                # Sleep briefly
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in chunk watcher: {e}")
                time.sleep(5)  # Wait longer on error
            
    def _download_and_segment(self, url: str, chunk_duration: int, output_dir: str, url_hash: str) -> None:
        """
        Download and segment the video using yt-dlp and ffmpeg.
        
        Args:
            url: YouTube URL to process
            chunk_duration: Duration of each chunk in seconds
            output_dir: Directory to store the chunks
            url_hash: Hash of the URL for tracking purposes
        """
        ytdlp_cmd = [
            'yt-dlp',
            '--quiet',
            '-o', '-',
            '--format', 'best',
            '--no-check-certificates',
            '--extractor-retries', '3',
            '--force-ipv4'
        ]
        
        # Add cookies if the file exists
        if os.path.exists(self.cookies_path):
            ytdlp_cmd.extend(['--cookies', self.cookies_path])
        else:
            # Handle development mode
            if hasattr(self, 'dev_mode') and self.dev_mode:
                self.logger.info("Development mode: Downloading without cookies")
            
        ytdlp_cmd.append(url)
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-hide_banner',     
            '-loglevel', 'error',  
            '-i', 'pipe:0',  
            '-c', 'copy',    
            '-f', 'segment',
            '-segment_time', str(chunk_duration),
            '-reset_timestamps', '1',
            '-segment_format_options', 'movflags=+faststart',  
            os.path.join(output_dir, 'chunk_%03d.mp4')
        ]
        
        # Start yt-dlp process
        self.logger.info(f"Starting download process for: {url}")
        
        ytdlp_proc = subprocess.Popen(
            ytdlp_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Store process for potential termination
        self.active_processes[url_hash] = ytdlp_proc
        
        # Start ffmpeg process
        try:
            self.logger.info("Starting ffmpeg segmentation process")
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=ytdlp_proc.stdout,
                stderr=subprocess.PIPE
            )
            
            # Store ffmpeg process as well 
            self.ffmpeg_processes[url_hash] = ffmpeg_proc
            
            # Start a thread to watch for new chunks
            watcher_thread = threading.Thread(
                target=self._watch_for_new_chunks,
                args=(output_dir, url_hash),
                daemon=True
            )
            watcher_thread.start()
            
            # Wait for process to complete or be terminated
            while True:
                # Check if stop was requested
                if url_hash in self.stop_signals and self.stop_signals[url_hash]:
                    self.logger.info(f"Stop signal received for {url}")
                    self._terminate_processes(ytdlp_proc, ffmpeg_proc)
                    break
                    
                # Check if processes are still running
                if ytdlp_proc.poll() is not None and ffmpeg_proc.poll() is not None:
                    # Both processes have completed
                    self.logger.info("Download and segmentation processes completed")
                    break
                    
                # Sleep briefly
                time.sleep(0.5)
            
            # Try to join the watcher thread with a timeout
            if watcher_thread.is_alive():
                watcher_thread.join(timeout=2.0)
            
            # Capture and log any stderr output
            ytdlp_stderr = ytdlp_proc.stderr.read().decode() if ytdlp_proc.stderr else ""
            ffmpeg_stderr = ffmpeg_proc.stderr.read().decode() if ffmpeg_proc.stderr else ""
            
            if ytdlp_stderr:
                self.logger.error(f"yt-dlp error: {ytdlp_stderr}")
                self.progress_trackers[url_hash].update(error=Exception(ytdlp_stderr))
                
            if ffmpeg_stderr:
                self.logger.error(f"FFmpeg error: {ffmpeg_stderr}")
                self.progress_trackers[url_hash].update(error=Exception(ffmpeg_stderr))
            
            # Check return codes
            if ytdlp_proc.returncode != 0 and not self.stop_signals.get(url_hash, False):
                raise subprocess.CalledProcessError(
                    ytdlp_proc.returncode, 
                    ytdlp_cmd, 
                    stderr=ytdlp_stderr
                )
                
            if ffmpeg_proc.returncode != 0 and not self.stop_signals.get(url_hash, False):
                raise subprocess.CalledProcessError(
                    ffmpeg_proc.returncode, 
                    ffmpeg_cmd, 
                    stderr=ffmpeg_stderr
                )
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error in video processing: {e}")
            self.logger.error(f"Command stderr: {e.stderr if hasattr(e, 'stderr') else 'No stderr'}")
            self.progress_trackers[url_hash].update(error=e)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.progress_trackers[url_hash].update(error=e)
            raise
        finally:
            # Clean up processes
            if url_hash in self.active_processes:
                del self.active_processes[url_hash]
            if url_hash in self.ffmpeg_processes:  
                del self.ffmpeg_processes[url_hash]
    
    def _terminate_processes(self, ytdlp_proc: subprocess.Popen, ffmpeg_proc: subprocess.Popen) -> bool:
        """
        Aggressively terminate yt-dlp and ffmpeg processes.
        
        Returns:
            bool: True if processes were terminated successfully
        """
        self.logger.info("Terminating download and segmentation processes")
        
        # Track if termination was successful
        all_terminated = True
        
        # Function to terminate a process based on the platform
        def terminate_process(proc, process_name):
            if proc is None or proc.poll() is not None:  # Skip if not running
                return True
                
            terminated = False
            
            try:
                if platform.system() == "Windows":
                    # On Windows use taskkill to force terminate
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], 
                                check=False, capture_output=True)
                    # Wait briefly to confirm termination
                    for _ in range(5):  # Wait up to 0.5 seconds
                        if proc.poll() is not None:
                            terminated = True
                            break
                        time.sleep(0.1)
                        
                    # If still running, try more drastic measures
                    if not terminated:
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], 
                                    check=False, capture_output=True)
                else:
                    # On Unix-like systems, try SIGTERM then SIGKILL
                    try:
                        proc.terminate()  # SIGTERM
                        try:
                            proc.wait(timeout=3)
                            terminated = True
                        except subprocess.TimeoutExpired:
                            self.logger.warning(f"{process_name} did not terminate gracefully, sending SIGKILL")
                            proc.kill()  # SIGKILL
                            proc.wait(timeout=3)
                            terminated = True
                    except Exception as e:
                        self.logger.error(f"Error terminating {process_name}: {e}")
                        terminated = False
            except Exception as e:
                self.logger.error(f"Error terminating {process_name} process: {e}")
                terminated = False
                
            return terminated
        
        # Try to terminate ffmpeg first (the child process)
        ffmpeg_terminated = terminate_process(ffmpeg_proc, "FFmpeg")
        
        # Then terminate yt-dlp
        ytdlp_terminated = terminate_process(ytdlp_proc, "yt-dlp")
        
        all_terminated = ffmpeg_terminated and ytdlp_terminated
        
        if all_terminated:
            self.logger.info("All processes successfully terminated")
        else:
            self.logger.warning("Some processes may still be running")
            
        return all_terminated
    
    def stop_processing(self, url: str) -> bool:
        """
        Stop an active processing job.
        
        Returns:
            bool: True if successfully stopped, False otherwise
        """
        url_hash = self._get_url_hash(url)
        
        if url_hash not in self.stop_signals:
            self.logger.warning(f"No active processing job found for {url}")
            return False
            
        self.logger.info(f"Setting stop signal for {url}")
        self.stop_signals[url_hash] = True
        
        # Get both processes
        ytdlp_proc = self.active_processes.get(url_hash)
        ffmpeg_proc = self.ffmpeg_processes.get(url_hash)
        
        # Terminate the processes
        if ytdlp_proc or ffmpeg_proc:
            self.logger.info(f"Terminating processes for {url}")
            terminated = self._terminate_processes(ytdlp_proc, ffmpeg_proc)
            
            if terminated:
                self.logger.info(f"Successfully terminated all processes for {url}")
                # Clean up
                if url_hash in self.active_processes:
                    del self.active_processes[url_hash]
                if url_hash in self.ffmpeg_processes:
                    del self.ffmpeg_processes[url_hash]
                return True
        
        # Wait for processes to terminate
        max_attempts = 10
        for i in range(max_attempts):
            # Check if processes are gone from dictionaries
            if url_hash not in self.active_processes and url_hash not in self.ffmpeg_processes:
                self.logger.info(f"Processes for {url} terminated successfully")
                return True
                
            # Check if processes have terminated
            ytdlp_terminated = ytdlp_proc is None or ytdlp_proc.poll() is not None
            ffmpeg_terminated = ffmpeg_proc is None or ffmpeg_proc.poll() is not None
            
            if ytdlp_terminated and ffmpeg_terminated:
                self.logger.info(f"All processes for {url} have terminated")
                # Clean up
                if url_hash in self.active_processes:
                    del self.active_processes[url_hash]
                if url_hash in self.ffmpeg_processes:
                    del self.ffmpeg_processes[url_hash]
                return True
                
            # Try again to terminate every few attempts
            if i % 3 == 2:
                self.logger.warning(f"Retrying process termination for {url}")
                self._terminate_processes(ytdlp_proc, ffmpeg_proc)
                
            time.sleep(1)
        
        # Last resort - forceful termination
        self.logger.warning(f"Attempting forceful termination for {url}")
        success = False
        
        try:
            if ytdlp_proc and ytdlp_proc.poll() is None:
                if platform.system() == "Windows":
                    subprocess.run(['taskkill', '/F', '/PID', str(ytdlp_proc.pid)], check=False)
                else:
                    os.kill(ytdlp_proc.pid, signal.SIGKILL)
                    
            if ffmpeg_proc and ffmpeg_proc.poll() is None:
                if platform.system() == "Windows":
                    subprocess.run(['taskkill', '/F', '/PID', str(ffmpeg_proc.pid)], check=False)
                else:
                    os.kill(ffmpeg_proc.pid, signal.SIGKILL)
                    
            # Clean up references regardless
            if url_hash in self.active_processes:
                del self.active_processes[url_hash]
            if url_hash in self.ffmpeg_processes:
                del self.ffmpeg_processes[url_hash]
                
            success = True
        except Exception as e:
            self.logger.error(f"Failed to forcefully terminate processes: {e}")
            success = False
            
        return success
    
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
        
    def cleanup_old_chunks(self, max_age_hours: int = 24) -> int:
        """
        Clean up old chunk files to free up disk space.
        
        Args:
            max_age_hours: Maximum age of chunks to keep in hours
            
        Returns:
            int: Number of files deleted
        """
        count = 0
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        try:
            # Walk through the data directory
            for root, dirs, files in os.walk(self.base_data_folder):
                for file in files:
                    if file.endswith('.mp4'):
                        file_path = os.path.join(root, file)
                        # Check file age
                        mtime = os.path.getmtime(file_path)
                        if mtime < cutoff_time:
                            os.remove(file_path)
                            count += 1
                            
            self.logger.info(f"Cleaned up {count} old chunk files")
        except Exception as e:
            self.logger.error(f"Error cleaning up old chunks: {e}")
            
        return count
    
    def check_cookie_configuration(self):
        """
        Check and log information about the cookie configuration.
        This helps diagnose cookie-related issues.
        
        Returns:
            dict: Information about cookie configuration
        """
        cookie_info = {
            "configured_path": self.cookies_path,
            "expanded_path": os.path.expanduser(self.cookies_path) if self.cookies_path else None,
            "file_exists": False,
            "file_size": 0,
            "file_permissions": None,
            "readable": False,
            "is_used": False
        }
        
        # First check if the cookies_path is set
        if not self.cookies_path:
            self.logger.warning("No cookies path configured")
            return cookie_info
            
        # Check if file exists
        if os.path.exists(self.cookies_path):
            cookie_info["file_exists"] = True
            cookie_info["file_size"] = os.path.getsize(self.cookies_path)
            cookie_info["is_used"] = True
            
            # Check permissions
            stats = os.stat(self.cookies_path)
            cookie_info["file_permissions"] = oct(stats.st_mode & 0o777)
            
            # Check if readable
            try:
                with open(self.cookies_path, 'r') as f:
                    first_line = f.readline().strip()
                    cookie_info["readable"] = True
                    cookie_info["first_line"] = first_line[:40] + "..." if len(first_line) > 40 else first_line
                
                self.logger.info(f"Cookie file found at {self.cookies_path}")
                self.logger.info(f"Cookie file size: {cookie_info['file_size']} bytes")
            except Exception as e:
                self.logger.warning(f"Cookie file exists but can't be read: {e}")
                cookie_info["error"] = f"File exists but can't be read: {str(e)}"
        else:
            # Check if we're in development mode
            if hasattr(self, 'dev_mode') and self.dev_mode:
                self.logger.info(f"Development mode: No cookie file at {self.cookies_path} (this is okay for local development)")
            else:
                self.logger.warning(f"Cookie file not found at {self.cookies_path}")
        
        return cookie_info
    
    def _test_url_accessibility(self, url: str, url_hash: str) -> bool:
        """
        Test if a YouTube URL is accessible using yt-dlp with graceful fallback.
        
        Args:
            url: YouTube URL to test
            url_hash: Hash of the URL for tracking
            
        Returns:
            bool: True if accessible, False otherwise
        """
        # Build command with basic options
        test_cmd = [
            'yt-dlp', 
            '--quiet', 
            '--simulate',
            '--no-check-certificates',
            '--extractor-retries', '3',
            '--force-ipv4'
        ]
        
        # Add cookies if the file exists
        if os.path.exists(self.cookies_path):
            test_cmd.extend(['--cookies', self.cookies_path])
            self.logger.info(f"Using cookies from {self.cookies_path}")
        else:
            if hasattr(self, 'dev_mode') and self.dev_mode:
                self.logger.info("Development mode: Proceeding without cookies")
            else:
                self.logger.warning(f"Cookies file not found at {self.cookies_path}, proceeding without cookies")
            
        test_cmd.append(url)
        
        try:
            self.logger.info(f"Testing URL accessibility: {url}")
            result = subprocess.run(test_cmd, check=True, capture_output=True, text=True)
            
            # Check if it's a live stream
            is_live = False
            if "is a live stream" in result.stdout:
                is_live = True
                self.logger.info(f"Detected live stream: {url}")
                
            self.progress_trackers[url_hash].update(is_live=is_live)
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error accessing URL with yt-dlp: {e.stderr}")
            self.progress_trackers[url_hash].update(error=e)
            self.progress_trackers[url_hash].status = "failed"
            
            # Check for specific errors related to missing cookies
            if "sign in to view" in e.stderr.lower() or "private video" in e.stderr.lower():
                if self.dev_mode:
                    self.logger.warning("Development mode: This video requires authentication. Consider adding cookies for production.")
                else:
                    raise ValueError(f"This video requires authentication. Please provide valid cookies.")
            else:
                raise ValueError(f"Could not access URL: {url}")
                
        except Exception as e:
            self.logger.error(f"Unexpected error testing URL accessibility: {e}")
            self.progress_trackers[url_hash].update(error=e)
            self.progress_trackers[url_hash].status = "failed"
            raise ValueError(f"Error testing URL accessibility: {e}")

# Example usage:
if __name__ == "__main__":
    import time
    
    def main():
        # Initialize the chunker
        chunker = YouTubeChunker()
        
        # Process a video
        try:
            output_path = chunker.process_video(
                url="https://www.youtube.com/watch?v=jfKfPfyJRdk",  # lofi hip hop radio - beats to relax/study to
                chunk_duration=30  # 30-second chunks
            )
            print(f"Video chunks being saved in: {output_path}")
            
            # Wait for a while and then stop the process
            time.sleep(60)
            chunker.stop_processing("https://www.youtube.com/watch?v=jfKfPfyJRdk")
            print("Processing stopped")
            
            # Check progress
            progress = chunker.get_progress("https://www.youtube.com/watch?v=jfKfPfyJRdk")
            print(f"Progress: {progress}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Run the example
    main()