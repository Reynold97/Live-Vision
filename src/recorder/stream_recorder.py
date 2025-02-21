import subprocess
import os
from datetime import datetime
import logging
from typing import Optional

class YouTubeChunker:
    """A class to download YouTube videos and split them into chunks."""
    
    def __init__(self, base_data_folder="data", cookies_path=None):
        """
        Initialize the YouTubeChunker.
        
        Args:
            base_data_folder (str): Base folder where all video chunks will be stored
            cookies_path (str): Path to the cookies file for YouTube authentication
        """
        self.base_data_folder = base_data_folder
        self.cookies_path = cookies_path or "/var/www/live-vision/cookies/youtube.txt"
        self._setup_logging()
        self._check_dependencies()
    
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
            subprocess.run(['yt-dlp', '--version'], capture_output=True)
            subprocess.run(['ffmpeg', '-version'], capture_output=True)
        except FileNotFoundError:
            raise SystemError(
                "Required dependencies not found. Please install yt-dlp and ffmpeg:\n"
                "pip install yt-dlp\n"
                "And install ffmpeg from your system package manager."
            )
    
    def _create_output_directory(self):
        """
        Create a nested directory structure for video chunks.
        Format: base_data_folder/YYYY_MM_DD/HH_MM/
        """
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
    
    def process_video(self, url: str, chunk_duration: int, output_dir: Optional[str] = None) -> str:
        """
        Download a YouTube video and split it into chunks.
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
            
            # Test yt-dlp can access the URL
            test_cmd = [
                'yt-dlp', 
                '--quiet', 
                '--simulate',
                '--no-check-certificates',
                '--extractor-retries', '3',
                '--force-ipv4',
                '--cookies', self.cookies_path,
                url
            ]
            try:
                subprocess.run(test_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error accessing URL with yt-dlp: {e.stderr.decode()}")
                raise ValueError(f"Could not access URL: {url}")
            
            self._download_and_segment(url, chunk_duration, output_dir)
            self.logger.info("Video processing completed successfully")
            return output_dir
                
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            raise
            
    def _download_and_segment(self, url: str, chunk_duration: int, output_dir: str) -> None:
        """Download and segment the video using yt-dlp and ffmpeg."""
        ytdlp_cmd = [
            'yt-dlp',
            '--quiet',
            '-o', '-',
            '--format', 'best',
            '--no-check-certificates',
            '--extractor-retries', '3',
            '--force-ipv4',
            '--cookies', self.cookies_path,
            url
        ]
        
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
        with subprocess.Popen(ytdlp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as ytdlp_proc:
            # Start ffmpeg process
            try:
                ffmpeg_proc = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=ytdlp_proc.stdout,
                    stderr=subprocess.PIPE
                )
                
                # Capture and log any stderr output
                stderr_data = ffmpeg_proc.stderr.read().decode()
                if stderr_data:
                    self.logger.error(f"FFmpeg error: {stderr_data}")
                
                # Wait for completion
                ffmpeg_proc.wait()
                if ffmpeg_proc.returncode != 0:
                    raise subprocess.CalledProcessError(
                        ffmpeg_proc.returncode, 
                        ffmpeg_cmd, 
                        stderr=stderr_data
                    )
                    
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error in video processing: {e}")
                self.logger.error(f"FFmpeg stderr: {e.stderr if hasattr(e, 'stderr') else 'No stderr'}")
                raise

# Example usage:
if __name__ == "__main__":
    # Initialize the chunker
    chunker = YouTubeChunker()
    
    # Process a video
    try:
        output_path = chunker.process_video(
            url="https://www.youtube.com/watch?v=IWBn0-KQIdI",
            chunk_duration=15  # 60-second chunks
        )
        print(f"Video chunks saved in: {output_path}")
    except Exception as e:
        print(f"Error: {e}")