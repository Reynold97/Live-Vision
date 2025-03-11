# src/recorder/m3u8_recorder.py
from typing import Optional, Dict, Any, List
from .base_recorder import BaseStreamRecorder, ChunkingProgress
import subprocess
import os
import hashlib
import threading
import time
import requests
import platform
import signal
import shlex

class M3U8Chunker(BaseStreamRecorder):
    """M3U8/HLS stream recorder implementation."""
    
    def __init__(self, base_data_folder="data"):
        """
        Initialize the M3U8Chunker.
        
        Args:
            base_data_folder (str): Base folder where all video chunks will be stored
        """
        super().__init__(base_data_folder)
        
        # Check ffmpeg dependency
        self._check_dependencies()
        
        # M3U8-specific initializations
        self.ffmpeg_processes = {}
        self.active_processes = {}
        self.progress_trackers = {}
        self.stop_signals = {}
        self.child_pids = {}  # Track child process IDs
        
    def _check_dependencies(self):
        """Check if required external programs are installed."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except FileNotFoundError:
            raise SystemError(
                "Required dependency not found. Please install ffmpeg from your system package manager."
            )
        except subprocess.CalledProcessError:
            raise SystemError(
                "Error running ffmpeg. Please ensure ffmpeg is properly installed."
            )
    
    def _test_url_accessibility(self, url: str, url_hash: str) -> bool:
        """
        Test if an M3U8 URL is accessible.
        
        Args:
            url: M3U8 URL to test
            url_hash: Hash of the URL for tracking
            
        Returns:
            bool: True if accessible, False otherwise
        """
        try:
            # Trim whitespace from URL
            url = url.strip()
            self.logger.info(f"Testing M3U8 URL accessibility: {url}")
            
            # Initialize progress tracker if needed
            if url_hash not in self.progress_trackers:
                self.progress_trackers[url_hash] = ChunkingProgress()
                
            # Try to fetch the m3u8 file
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Check if content appears to be an m3u8 file
            content = response.text.lower()
            
            if not ('#extm3u' in content or '.ts' in content or '#ext-x-stream-inf' in content):
                raise ValueError("URL does not appear to be a valid M3U8 stream")
                
            # Determine if this is a live stream by looking for live indicators
            is_live = '#ext-x-endlist' not in content
            self.progress_trackers[url_hash].update(is_live=is_live)
            
            if is_live:
                self.logger.info(f"Detected live M3U8 stream: {url}")
            else:
                self.logger.info(f"Detected VOD M3U8 stream: {url}")
                
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error accessing M3U8 URL: {e}")
            self.progress_trackers[url_hash].update(error=e)
            self.progress_trackers[url_hash].status = "failed"
            raise ValueError(f"Could not access M3U8 URL: {url}. Error: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error testing M3U8 URL accessibility: {e}")
            self.progress_trackers[url_hash].update(error=e)
            self.progress_trackers[url_hash].status = "failed"
            raise ValueError(f"Error testing M3U8 URL accessibility: {e}")
    
    def process_video(self, url: str, chunk_duration: int = 30, output_dir: Optional[str] = None) -> str:
        """
        Process an M3U8 stream into chunks.
        
        Args:
            url: M3U8 URL to process (must be trimmed)
            chunk_duration: Duration of each chunk in seconds
            output_dir: Optional custom output directory
            
        Returns:
            str: Path to the output directory
        """
        try:
            # Clean the URL to avoid issues with trailing spaces
            url = url.strip()
            
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
            self.child_pids[url_hash] = []
            
            # Test accessibility
            is_accessible = self._test_url_accessibility(url, url_hash)
            if not is_accessible:
                raise ValueError(f"Could not access URL: {url}")
            
            # Start downloading and segmenting
            self.progress_trackers[url_hash].status = "running"
            self._download_and_segment_m3u8(url, chunk_duration, output_dir, url_hash)
            
            # If we get here, process completed successfully
            self.progress_trackers[url_hash].status = "completed"
            self.logger.info("M3U8 stream processing completed successfully")
            return output_dir
                
        except Exception as e:
            url_hash = self._get_url_hash(url)
            if url_hash in self.progress_trackers:
                self.progress_trackers[url_hash].update(error=e)
                self.progress_trackers[url_hash].status = "failed"
            self.logger.error(f"Error processing M3U8 stream: {e}")
            raise
            
    def _download_and_segment_m3u8(self, url: str, chunk_duration: int, output_dir: str, url_hash: str) -> None:
        """
        Download and segment M3U8 stream using ffmpeg.
        
        Args:
            url: M3U8 URL to process
            chunk_duration: Duration of each chunk in seconds
            output_dir: Directory to store the chunks
            url_hash: Hash of the URL for tracking purposes
        """
        # ffmpeg command to download and segment the stream
        ffmpeg_cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'warning',
            '-i', url,
            '-c', 'copy',  # Copy without re-encoding
            '-f', 'segment',
            '-segment_time', str(chunk_duration),
            '-reset_timestamps', '1',
            '-segment_format_options', 'movflags=+faststart',
            os.path.join(output_dir, 'chunk_%03d.mp4')
        ]
        
        # Start ffmpeg process
        self.logger.info(f"Starting ffmpeg process for M3U8 stream: {url}")
        self.logger.debug(f"Command: {' '.join(ffmpeg_cmd)}")
        
        try:
            # Create process with appropriate settings for platform
            if platform.system() == "Windows":
                # On Windows, create a new process group to enable sending CTRL_BREAK_EVENT
                ffmpeg_proc = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                # On Unix-like systems, create a new session to enable killing process group
                ffmpeg_proc = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid  # Create a new session
                )
            
            # Store process for potential termination
            self.active_processes[url_hash] = ffmpeg_proc
            self.ffmpeg_processes[url_hash] = ffmpeg_proc
            
            # Store child process PIDs if possible
            self._find_child_processes(url_hash, ffmpeg_proc.pid)
            
            # Start a thread to watch for new chunks
            watcher_thread = threading.Thread(
                target=self._watch_for_new_chunks,
                args=(output_dir, url_hash),
                daemon=True
            )
            watcher_thread.start()
            
            # Start a monitoring thread to periodically check for new child processes
            monitoring_thread = threading.Thread(
                target=self._monitor_child_processes,
                args=(url_hash, ffmpeg_proc.pid),
                daemon=True
            )
            monitoring_thread.start()
            
            # Wait for process to complete or be terminated
            while True:
                # Check if stop was requested
                if self.stop_signals.get(url_hash, False):
                    self.logger.info(f"Stop signal received for {url}")
                    self._terminate_ffmpeg_process(url_hash, ffmpeg_proc)
                    break
                    
                # Check if process is still running
                if ffmpeg_proc.poll() is not None:
                    # Process has completed
                    self.logger.info("ffmpeg process completed")
                    break
                    
                # Sleep briefly
                time.sleep(0.5)
                
            # Try to join the watcher thread with a timeout
            if watcher_thread.is_alive():
                watcher_thread.join(timeout=2.0)
                
            # Capture and log any stderr output
            ffmpeg_stderr = ffmpeg_proc.stderr.read().decode() if ffmpeg_proc.stderr else ""
            
            if ffmpeg_stderr:
                self.logger.error(f"ffmpeg error: {ffmpeg_stderr}")
                self.progress_trackers[url_hash].update(error=Exception(ffmpeg_stderr))
            
            # Check return code if not stopped manually
            if ffmpeg_proc.returncode != 0 and not self.stop_signals.get(url_hash, False):
                raise subprocess.CalledProcessError(
                    ffmpeg_proc.returncode,
                    ffmpeg_cmd,
                    stderr=ffmpeg_stderr
                )
                
        except Exception as e:
            self.logger.error(f"Error processing M3U8 stream: {e}")
            self.progress_trackers[url_hash].update(error=e)
            raise
            
        finally:
            # Clean up processes
            if url_hash in self.active_processes:
                del self.active_processes[url_hash]
            if url_hash in self.ffmpeg_processes:
                del self.ffmpeg_processes[url_hash]
    
    def _find_child_processes(self, url_hash: str, parent_pid: int) -> None:
        """
        Find and store child process IDs for a parent process.
        Uses platform-specific methods.
        
        Args:
            url_hash: Hash of the URL for tracking
            parent_pid: PID of the parent process
        """
        try:
            if platform.system() == "Windows":
                # On Windows, use WMIC to find child processes
                cmd = f'wmic process where (ParentProcessId={parent_pid}) get ProcessId'
                output = subprocess.check_output(cmd, shell=True).decode()
                
                # Parse output to get PIDs
                lines = output.strip().split('\n')
                if len(lines) > 1:  # First line is header
                    for line in lines[1:]:
                        if line.strip().isdigit():
                            child_pid = int(line.strip())
                            if child_pid not in self.child_pids.get(url_hash, []):
                                self.child_pids.setdefault(url_hash, []).append(child_pid)
                                self.logger.debug(f"Found child process: {child_pid} for parent: {parent_pid}")
            else:
                # On Unix-like systems, use ps to find child processes
                cmd = f'ps -o pid --ppid {parent_pid} --no-headers'
                try:
                    output = subprocess.check_output(cmd, shell=True).decode()
                    for line in output.strip().split('\n'):
                        if line.strip():
                            child_pid = int(line.strip())
                            if child_pid not in self.child_pids.get(url_hash, []):
                                self.child_pids.setdefault(url_hash, []).append(child_pid)
                                self.logger.debug(f"Found child process: {child_pid} for parent: {parent_pid}")
                except subprocess.CalledProcessError:
                    # Process may have terminated
                    pass
        except Exception as e:
            self.logger.warning(f"Failed to find child processes: {e}")
    
    def _monitor_child_processes(self, url_hash: str, parent_pid: int) -> None:
        """
        Periodically monitor and update the list of child processes.
        
        Args:
            url_hash: Hash of the URL for tracking
            parent_pid: PID of the parent process
        """
        while (
            url_hash in self.progress_trackers and 
            self.progress_trackers[url_hash].status == "running" and
            not self.stop_signals.get(url_hash, True)
        ):
            try:
                self._find_child_processes(url_hash, parent_pid)
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.logger.error(f"Error monitoring child processes: {e}")
                time.sleep(10)  # Longer delay on error
            
    def _watch_for_new_chunks(self, output_dir: str, url_hash: str) -> None:
        """
        Watch for new chunks being created in the output directory.
        
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
        
    def stop_processing(self, url: str) -> bool:
        """
        Stop an active processing job.
        
        Args:
            url: URL of the stream to stop processing
            
        Returns:
            bool: True if successfully stopped, False otherwise
        """
        # Clean the URL to avoid issues with trailing spaces
        url = url.strip()
        url_hash = self._get_url_hash(url)
        
        if url_hash not in self.stop_signals:
            self.logger.warning(f"No active processing job found for {url}")
            return False
            
        self.logger.info(f"Setting stop signal for {url}")
        self.stop_signals[url_hash] = True
        
        # Get ffmpeg process
        ffmpeg_proc = self.ffmpeg_processes.get(url_hash)
        
        if not ffmpeg_proc:
            self.logger.warning(f"No active ffmpeg process found for {url}")
            return self._cleanup_references(url_hash)
        
        # Terminate ffmpeg process
        self.logger.info(f"Terminating ffmpeg process for {url}")
        terminated = self._terminate_ffmpeg_process(url_hash, ffmpeg_proc)
        
        # Verify termination
        if terminated:
            self.logger.info(f"Successfully terminated ffmpeg process for {url}")
            return self._cleanup_references(url_hash)
        
        # Termination might have failed, wait and retry
        max_attempts = 5
        for i in range(max_attempts):
            self.logger.warning(f"Termination attempt {i+1}/{max_attempts}")
            
            # Check if process is still running
            if ffmpeg_proc.poll() is not None:
                self.logger.info(f"Process for {url} has terminated after {i+1} attempts")
                return self._cleanup_references(url_hash)
                
            # Try again with more aggressive methods
            self._kill_process_aggressive(url_hash, ffmpeg_proc)
            
            # Wait before checking again
            time.sleep(2)
        
        # Last resort - kill all ffmpeg processes (dangerous!)
        self.logger.warning(f"All termination attempts failed, trying to kill all ffmpeg processes")
        self._kill_all_ffmpeg_processes()
        
        # Clean up references regardless of success
        return self._cleanup_references(url_hash)
    
    def _cleanup_references(self, url_hash: str) -> bool:
        """
        Clean up all references to processes.
        
        Args:
            url_hash: Hash of the URL for tracking
            
        Returns:
            bool: True if cleanup was successful
        """
        try:
            if url_hash in self.active_processes:
                del self.active_processes[url_hash]
            if url_hash in self.ffmpeg_processes:
                del self.ffmpeg_processes[url_hash]
            if url_hash in self.child_pids:
                del self.child_pids[url_hash]
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up references: {e}")
            return False
        
    def _terminate_ffmpeg_process(self, url_hash: str, proc: subprocess.Popen) -> bool:
        """
        Terminate an ffmpeg process with all known methods.
        
        Args:
            url_hash: Hash of the URL for tracking
            proc: Process to terminate
            
        Returns:
            bool: True if terminated successfully, False otherwise
        """
        if proc is None or proc.poll() is not None:
            return True
            
        terminated = False
        
        try:
            # Try the most gentle method first
            if platform.system() == "Windows":
                # On Windows, try CTRL_BREAK_EVENT
                try:
                    self.logger.info("Sending CTRL_BREAK_EVENT signal")
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                    time.sleep(1)
                except Exception as e:
                    self.logger.warning(f"Failed to send CTRL_BREAK_EVENT: {e}")
                    
                # If still running, try taskkill (gentle)
                if proc.poll() is None:
                    self.logger.info("Sending taskkill signal")
                    subprocess.run(['taskkill', '/PID', str(proc.pid)], 
                                  check=False, capture_output=True)
                    time.sleep(1)
                    
                # If still running, use taskkill /F (forceful)
                if proc.poll() is None:
                    self.logger.info("Sending forceful taskkill signal")
                    subprocess.run(['taskkill', '/F', '/PID', str(proc.pid)], 
                                  check=False, capture_output=True)
                    time.sleep(1)
                    
                # If still running, try taskkill /T to kill the tree
                if proc.poll() is None:
                    self.logger.info("Sending tree-kill taskkill signal")
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], 
                                  check=False, capture_output=True)
                    time.sleep(1)
            else:
                # On Unix, try SIGTERM first
                try:
                    self.logger.info("Sending SIGTERM signal")
                    proc.terminate()
                    time.sleep(1)
                except Exception as e:
                    self.logger.warning(f"Failed to send SIGTERM: {e}")
                    
                # If still running, try SIGKILL
                if proc.poll() is None:
                    self.logger.info("Sending SIGKILL signal")
                    proc.kill()
                    time.sleep(1)
                    
                # If still running, try killing the process group
                if proc.poll() is None:
                    self.logger.info("Killing process group")
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError) as e:
                        self.logger.warning(f"Failed to kill process group: {e}")
            
            # Kill child processes
            child_pids = self.child_pids.get(url_hash, [])
            if child_pids:
                self.logger.info(f"Killing {len(child_pids)} child processes")
                for child_pid in child_pids:
                    try:
                        if platform.system() == "Windows":
                            subprocess.run(['taskkill', '/F', '/PID', str(child_pid)], 
                                          check=False, capture_output=True)
                        else:
                            os.kill(child_pid, signal.SIGKILL)
                    except Exception as e:
                        self.logger.warning(f"Failed to kill child process {child_pid}: {e}")
            
            # Check if process terminated
            terminated = proc.poll() is not None
            
            if terminated:
                self.logger.info("Successfully terminated ffmpeg process")
            else:
                self.logger.warning("Failed to terminate ffmpeg process through normal methods")
                
            return terminated
            
        except Exception as e:
            self.logger.error(f"Error during process termination: {e}")
            return False
    
    def _kill_process_aggressive(self, url_hash: str, proc: subprocess.Popen) -> None:
        """
        Use more aggressive methods to kill a process.
        
        Args:
            url_hash: Hash of the URL for tracking
            proc: Process to kill
        """
        if proc is None or proc.poll() is not None:
            return
            
        try:
            # Try different methods based on platform
            if platform.system() == "Windows":
                # On Windows, try even more aggressive methods
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], 
                              check=False, capture_output=True)
                
                # Also try taskkill by image name
                subprocess.run(['taskkill', '/F', '/IM', 'ffmpeg.exe'], 
                              check=False, capture_output=True)
            else:
                # On Unix, try various kill methods
                try:
                    # Kill entire process group
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    pass
                    
                # Direct kill with SIGKILL
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                    
                # Try pkill command as a last resort
                try:
                    subprocess.run(['pkill', '-9', '-P', str(proc.pid)], 
                                  check=False, capture_output=True)
                    subprocess.run(['pkill', '-9', 'ffmpeg'], 
                                  check=False, capture_output=True)
                except Exception:
                    pass
                    
            # Kill all known child processes
            for child_pid in self.child_pids.get(url_hash, []):
                try:
                    if platform.system() == "Windows":
                        subprocess.run(['taskkill', '/F', '/PID', str(child_pid)], 
                                      check=False, capture_output=True)
                    else:
                        os.kill(child_pid, signal.SIGKILL)
                except Exception:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error during aggressive process killing: {e}")
    
    def _kill_all_ffmpeg_processes(self) -> None:
        """Kill all ffmpeg processes on the system (dangerous! Use as last resort)."""
        try:
            if platform.system() == "Windows":
                # On Windows, kill all ffmpeg.exe processes
                subprocess.run(['taskkill', '/F', '/IM', 'ffmpeg.exe'], 
                              check=False, capture_output=True)
            else:
                # On Unix, use pkill
                subprocess.run(['pkill', '-9', 'ffmpeg'], 
                              check=False, capture_output=True)
                
            self.logger.warning("Attempted to kill all ffmpeg processes")
        except Exception as e:
            self.logger.error(f"Error killing all ffmpeg processes: {e}")
            
    def get_progress(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get progress information for a URL.
        
        Args:
            url: URL to get progress for
            
        Returns:
            Dict or None: Progress information or None if not found
        """
        url = url.strip()
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