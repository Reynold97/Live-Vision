import subprocess
import os
from datetime import datetime
import logging
from typing import Optional

class YouTubeChunker:
    """A class to download YouTube videos and split them into chunks."""
    
    def __init__(self, base_data_folder="data"):
        """
        Initialize the YouTubeChunker.
        
        Args:
            base_data_folder (str): Base folder where all video chunks will be stored
        """
        self.base_data_folder = base_data_folder
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
        Note: This method is designed to be run in a separate thread via asyncio.to_thread
        
        Args:
            url (str): YouTube video URL
            chunk_duration (int): Duration of each chunk in seconds
            output_dir (str, optional): Directory to save chunks. If None, creates based on date/time
            
        Returns:
            str: Path to the directory containing the video chunks
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = self._create_output_directory()
            
        self.logger.info(f"Created output directory: {output_dir}")
        
        try:
            self._download_and_segment(url, chunk_duration, output_dir)
            self.logger.info("Video processing completed successfully")
            return output_dir
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error processing video: {e}")
            raise
            
    def _download_and_segment(self, url: str, chunk_duration: int, output_dir: str) -> None:
        """Download and segment the video using yt-dlp and ffmpeg."""
        ytdlp_cmd = [
            'yt-dlp',
            '--quiet',  # Suppress yt-dlp output
            '-o', '-',  
            '--format', 'best',  
            url
        ]
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-hide_banner',     # Hide ffmpeg compilation details
            '-loglevel', 'error',  # Only show errors
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
                
                # Monitor ffmpeg stderr for segment creation
                while True:
                    line = ffmpeg_proc.stderr.readline()
                    if not line:
                        break
                    
                    # Log only when a new segment starts
                    if b'Opening' in line and b'.mp4' in line:
                        chunk_name = line.decode().split("'")[1].split('\\')[-1]
                        self.logger.info(f"Creating new chunk: {chunk_name}")
                
                # Wait for completion
                ffmpeg_proc.wait()
                if ffmpeg_proc.returncode != 0:
                    raise subprocess.CalledProcessError(ffmpeg_proc.returncode, ffmpeg_cmd)
                    
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error in video processing: {e}")
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