"""
Cookie testing script for Live-Vision

This script helps verify YouTube cookie configuration for the Live-Vision application.
It checks for cookie files in the expected locations and tests them against YouTube.

Usage:
  python test_cookies.py [test_url]

If no URL is provided, a default public video will be used.
"""

import os
import subprocess
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default test video (public)
DEFAULT_TEST_URL = "https://www.youtube.com/watch?v=jfKfPfyJRdk"

def check_environment():
    """Check the execution environment."""
    env_mode = os.getenv("ENVIRONMENT", "").lower()
    logger.info(f"Current environment mode: {env_mode or 'Not set'}")
    
    if env_mode == "development":
        logger.info("Running in development mode: Cookies are optional")
    elif env_mode == "production":
        logger.info("Running in production mode: Cookies are recommended")
    else:
        logger.info("No environment mode set: Defaulting to standard behavior")
    
    return env_mode

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        # Check yt-dlp version
        ytdlp_version = subprocess.run(
            ['yt-dlp', '--version'], 
            capture_output=True, 
            text=True, 
            check=True
        ).stdout.strip()
        logger.info(f"Found yt-dlp version: {ytdlp_version}")
        
        # Check ffmpeg
        ffmpeg_result = subprocess.run(
            ['ffmpeg', '-version'], 
            capture_output=True, 
            text=True, 
            check=True
        ).stdout.split('\n')[0]
        logger.info(f"Found ffmpeg: {ffmpeg_result}")
        
        return True
    except FileNotFoundError as e:
        logger.error(f"Required dependency not found: {e}")
        logger.error("Please install yt-dlp and ffmpeg")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running dependency check: {e}")
        return False

def find_cookie_files():
    """Find potential YouTube cookie files in common locations."""
    cookie_paths = []
    
    # Common paths to check
    paths_to_check = [
        os.path.join(os.path.expanduser("~"), "cookies", "youtube.txt"),  # User's home dir
        "/root/cookies/youtube.txt",  # Root's home dir
        os.path.join(os.getcwd(), "cookies", "youtube.txt"),  # Current dir
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            cookie_paths.append(path)
            
    return cookie_paths

def check_cookie_file(cookie_path):
    """Analyze a cookie file for basic validity."""
    try:
        file_size = os.path.getsize(cookie_path)
        logger.info(f"Cookie file at {cookie_path}:")
        logger.info(f"  - Size: {file_size} bytes")
        
        # Check if the file size is reasonable
        if file_size < 100:
            logger.warning("  - File seems too small to contain valid cookies")
        
        # Check file permissions
        permissions = oct(os.stat(cookie_path).st_mode & 0o777)
        logger.info(f"  - Permissions: {permissions}")
        
        if permissions != "0o600" and permissions != "0o400":
            logger.warning("  - Recommended permissions are 600 (read/write for owner only)")
        
        # Check file content format
        with open(cookie_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line == "# Netscape HTTP Cookie File":
                logger.info("  - Format: Valid Netscape cookie format")
            else:
                logger.warning(f"  - Format: Unexpected format, first line: '{first_line}'")
            
            # Check for important cookie names
            content = f.read()
            important_cookies = ['SID', 'HSID', 'SSID', 'APISID', 'SAPISID', 'LOGIN_INFO']
            found_cookies = []
            
            for cookie in important_cookies:
                if cookie in content:
                    found_cookies.append(cookie)
            
            if found_cookies:
                logger.info(f"  - Found important cookies: {', '.join(found_cookies)}")
                missing = set(important_cookies) - set(found_cookies)
                if missing:
                    logger.warning(f"  - Missing important cookies: {', '.join(missing)}")
            else:
                logger.warning("  - No important authentication cookies found")
        
        return True
    except Exception as e:
        logger.error(f"Error checking cookie file {cookie_path}: {e}")
        return False

def test_youtube_access(cookie_path=None, url=DEFAULT_TEST_URL):
    """Test YouTube access with or without cookies."""
    cmd = ['yt-dlp', '--simulate', '--quiet', '-F', url]
    
    if cookie_path:
        cmd.extend(['--cookies', cookie_path])
        logger.info(f"Testing YouTube access WITH cookies at {cookie_path}")
    else:
        logger.info("Testing YouTube access WITHOUT cookies")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        format_count = result.stdout.count('\n')
        
        if format_count > 0:
            logger.info(f"SUCCESS: Access successful, found {format_count} formats")
            return True
        else:
            logger.warning("WARNING: Command succeeded but no formats were found")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"ERROR: YouTube access failed: {e.stderr}")
        
        # Check for common errors
        if "Private video" in e.stderr:
            logger.error("This appears to be a private video that requires authentication")
        elif "Sign in" in e.stderr:
            logger.error("This video requires sign-in to access")
        elif "not available in your country" in e.stderr:
            logger.error("This video is region-restricted")
            
        return False

def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("Live-Vision YouTube Cookie Testing Tool")
    logger.info("=" * 60)
    
    # Get the test URL
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
        logger.info(f"Using provided URL: {test_url}")
    else:
        test_url = DEFAULT_TEST_URL
        logger.info(f"Using default URL: {test_url}")
    
    # Check environment
    env_mode = check_environment()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Please install required tools.")
        return 1
    
    # Find cookie files
    cookie_files = find_cookie_files()
    if cookie_files:
        logger.info(f"Found {len(cookie_files)} potential cookie files")
    else:
        logger.warning("No cookie files found in common locations")
        
        # Check if we should create directories
        home_cookie_dir = os.path.join(os.path.expanduser("~"), "cookies")
        if not os.path.exists(home_cookie_dir):
            logger.info(f"The directory {home_cookie_dir} does not exist")
            
            create_dir = input("Would you like to create it? (y/n): ").lower() == 'y'
            if create_dir:
                try:
                    os.makedirs(home_cookie_dir, exist_ok=True)
                    logger.info(f"Created directory: {home_cookie_dir}")
                except Exception as e:
                    logger.error(f"Error creating directory: {e}")
    
    # Test YouTube access without cookies first
    logger.info("\nTesting YouTube access WITHOUT cookies:")
    no_cookie_access = test_youtube_access(url=test_url)
    
    # Now test each cookie file if found
    cookie_success = False
    if cookie_files:
        for cookie_path in cookie_files:
            logger.info(f"\nAnalyzing cookie file: {cookie_path}")
            check_cookie_file(cookie_path)
            
            logger.info("\nTesting YouTube access WITH this cookie file:")
            if test_youtube_access(cookie_path, test_url):
                cookie_success = True
    
    # Provide summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    if no_cookie_access:
        logger.info("✅ YouTube access works WITHOUT cookies")
        logger.info("   This likely means the test video is publicly accessible")
        
        if env_mode == "development":
            logger.info("   This is fine for development mode with public videos")
        else:
            logger.info("   For private or restricted videos, you'll still need cookies")
    else:
        logger.info("❌ YouTube access FAILED without cookies")
        
        if cookie_success:
            logger.info("   But succeeded with cookies - this is normal for private videos")
        else:
            logger.info("   This may indicate network issues or YouTube restrictions")
    
    if cookie_files:
        if cookie_success:
            logger.info("✅ At least one cookie file worked successfully")
        else:
            logger.info("❌ None of the cookie files granted access")
            logger.info("   Your cookies may be expired or invalid")
    else:
        logger.info("⚠️ No cookie files were found to test")
        logger.info("   Create a cookies/youtube.txt file for authenticated access")
    
    if env_mode == "development":
        logger.info("\nDevelopment mode is active: Cookie authentication is optional")
        logger.info("The application will work with public videos without cookies")
    elif env_mode == "production":
        logger.info("\nProduction mode is active: Cookie authentication is recommended")
        if not cookie_success and not no_cookie_access:
            logger.warning("⚠️ You may encounter access issues with some videos in production mode")
    
    return 0 if (no_cookie_access or cookie_success) else 1

if __name__ == "__main__":
    sys.exit(main())