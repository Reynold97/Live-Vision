#!/usr/bin/env python3
"""
Test script to verify YouTube cookie functionality with yt-dlp.
Save this in your project root and run it to test cookie functionality.
"""

import os
import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_cookie_file(cookie_path):
    """Check if cookie file exists and has proper format."""
    logger.info(f"Checking cookie file at: {cookie_path}")
    
    if not os.path.exists(cookie_path):
        logger.error(f"Cookie file not found at: {cookie_path}")
        return False
    
    file_size = os.path.getsize(cookie_path)
    logger.info(f"Cookie file size: {file_size} bytes")
    
    if file_size < 100:
        logger.warning("Cookie file seems too small, might not contain enough data")
    
    try:
        with open(cookie_path, 'r') as f:
            first_line = f.readline().strip()
            if not first_line.startswith("# Netscape HTTP Cookie File"):
                logger.warning("Cookie file doesn't start with expected Netscape format header")
            
            # Check for some common cookie names
            content = f.read()
            important_cookies = ['SID', 'HSID', 'SSID', 'APISID', 'SAPISID', 'LOGIN_INFO']
            found_cookies = []
            
            for cookie in important_cookies:
                if cookie in content:
                    found_cookies.append(cookie)
            
            if found_cookies:
                logger.info(f"Found important cookies: {', '.join(found_cookies)}")
            else:
                logger.warning("No important authentication cookies found")
    
    except Exception as e:
        logger.error(f"Error reading cookie file: {e}")
        return False
    
    return True

def test_ytdlp_with_cookies(cookie_path, test_url="https://www.youtube.com/watch?v=jfKfPfyJRdk"):
    """Test yt-dlp with the cookie file."""
    logger.info(f"Testing yt-dlp with cookie file: {cookie_path}")
    logger.info(f"Test URL: {test_url}")
    
    try:
        # Run yt-dlp with cookies to list formats (no actual download)
        cmd = ['yt-dlp', '--cookies', cookie_path, '-F', test_url]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            logger.info("yt-dlp test SUCCESSFUL!")
            logger.info(f"Found {result.stdout.count('audio only') + result.stdout.count('video only')} formats")
            return True
        else:
            logger.error(f"yt-dlp test FAILED with exit code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error running yt-dlp test: {e}")
        return False

def main():
    """Main function to test cookie functionality."""
    # Default paths to check
    paths_to_check = [
        os.path.join(os.path.expanduser("~"), "cookies", "youtube.txt"),  # User's home
        "/root/cookies/youtube.txt",  # Root's home
        "./cookies/youtube.txt"  # Current directory
    ]
    
    logger.info("YouTube Cookie Test Script")
    logger.info("=========================")
    
    # Check yt-dlp version
    try:
        version_result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True, check=True)
        logger.info(f"yt-dlp version: {version_result.stdout.strip()}")
    except Exception as e:
        logger.error(f"Error checking yt-dlp version: {e}")
        logger.error("Please make sure yt-dlp is installed and accessible")
        return 1
    
    # Check each path
    for path in paths_to_check:
        logger.info(f"\nChecking path: {path}")
        if check_cookie_file(path):
            logger.info(f"Cookie file exists at {path}, testing with yt-dlp...")
            if test_ytdlp_with_cookies(path):
                logger.info(f"SUCCESS: Cookie file at {path} works with yt-dlp!")
                return 0
        else:
            logger.info(f"No valid cookie file at {path}")
    
    logger.error("Could not find a working cookie file at any expected location")
    logger.info("Please create a cookies directory in your home folder and add youtube.txt")
    return 1

if __name__ == "__main__":
    sys.exit(main())