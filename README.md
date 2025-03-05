# Live-Vision: YouTube Stream Analysis Platform

Live-Vision is a powerful application for real-time YouTube stream analysis using Google's Gemini Vision API. The platform segments YouTube videos and livestreams into chunks, analyzes the visual content with advanced AI, and displays the results in a responsive web interface.

## Features

- **Real-time YouTube Stream Analysis**: Process and analyze YouTube videos and livestreams in real-time
- **Multi-Pipeline Processing**: Run multiple analysis pipelines concurrently
- **Advanced AI Vision Analysis**: Leverage Google's Gemini Vision API for deep visual understanding
- **WebSocket-Based Real-Time Updates**: Get analysis results as they become available
- **Configurable Analysis Parameters**: Customize chunk duration, analysis prompts, and more
- **Web Search Enhancement**: Option to use web search to enrich analysis results
- **Environment-Aware Configuration**: Different setups for development and production
- **Cookie-Based Authentication**: Access private or restricted YouTube content

## Architecture

The project consists of two main components:

### Backend (Python/FastAPI)

- **Core**: Configuration, state management, and pipeline coordination
- **Recorder**: YouTube video downloading and segmentation using yt-dlp and FFmpeg
- **Analyzer**: Integration with Google's Gemini Vision API
- **API**: FastAPI endpoints and WebSocket communication

### Frontend (React)

- **Modern UI**: Clean, responsive dashboard for controlling and monitoring analyses
- **Real-time Updates**: Live analysis results via WebSocket connection
- **Pipeline Visualization**: Visual representation of pipeline states and progress
- **Advanced Controls**: Customizable analysis options and settings

## Project Structure

```
/opt/live-vision/
├── env/                  # Python virtual environment
├── src/                  # Backend source code
│   ├── analyzer/         # Gemini Vision analysis
│   ├── api/              # FastAPI endpoints
│   ├── core/             # Core configuration and management
│   ├── pipeline/         # Video processing pipeline
│   └── recorder/         # YouTube stream handling
├── frontend/            
│   ├── dist/             # Built frontend files
│   └── node_modules/     # Frontend dependencies
├── data/                 # Video chunks directory
├── cookies/              # YouTube cookies for authentication
├── .env                  # Environment variables
└── requirements.txt      # Python dependencies
```

## Prerequisites

- Python 3.9 or later
- Node.js 18 or later
- FFmpeg
- yt-dlp
- Google Gemini API key

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Reynold97/Live-Vision.git
cd Live-Vision
```

### 2. Backend Setup

```bash
# Create and activate virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```
# Required
GOOGLE_API_KEY=your_gemini_api_key

# Optional but recommended
ENVIRONMENT=development  # or 'production' for production deployments

# Pipeline settings
MAX_CONCURRENT_PIPELINES=3
DEFAULT_CHUNK_DURATION=30
MIN_CHUNK_DURATION=10
MAX_CHUNK_DURATION=300

# Analysis settings
USE_WEB_SEARCH=true
EXPORT_RESPONSES=true
```

## YouTube Authentication Setup

The application can access private or region-restricted YouTube content using cookies from an authenticated browser session.

### Development Mode

In development mode, the application works with public videos without requiring cookie authentication. Set `ENVIRONMENT=development` in your `.env` file for this behavior.

### Production Mode

For production deployments or to access private content:

1. **Extract Cookies from Your Browser**:
   - Install a browser extension like "Get cookies.txt" ([Chrome](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) or [Firefox](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/))
   - Go to YouTube and log in to your account
   - Use the extension to export cookies for youtube.com
   - Save the file as `youtube.txt`

2. **Add Cookies to the Application**:
   - Create a `cookies` directory in your home folder (`~/cookies` or `/root/cookies`)
   - Place the `youtube.txt` file in this directory
   - Ensure proper permissions: `chmod 600 ~/cookies/youtube.txt`

3. **Configure for Production**:
   - Set `ENVIRONMENT=production` in your `.env` file

## Running the Application

### Development Mode

```bash
# Start the backend
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# In a separate terminal, start the frontend dev server
cd frontend
npm run dev
```

### Production Mode

```bash
# Start the backend
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2

# Or using systemd service (see deployment guide)
sudo systemctl start live-vision
```

## Usage

1. **Access the Dashboard**:
   - Open your browser to `http://localhost:8000` (development) or your server address (production)

2. **Start an Analysis**:
   - Enter a YouTube video or livestream URL
   - Set the chunk duration (in seconds)
   - Configure advanced options if needed
   - Click "Start Analysis"

3. **Monitor Pipelines**:
   - View active pipelines in the dashboard
   - Track progress and status of each pipeline
   - Stop pipelines as needed

4. **View Analysis Results**:
   - Analysis results appear in real-time as each video chunk is processed
   - Results include detailed AI-generated descriptions and observations

## Troubleshooting

### YouTube Access Issues

If you're having trouble accessing YouTube content:

1. **Check Cookie Configuration**:
   - Verify the cookie file exists in the correct location
   - Make sure the cookie file has the correct format
   - Ensure your YouTube cookies are not expired

2. **Development vs Production**:
   - Confirm your environment setting matches your needs
   - Development mode works best with public videos
   - Production mode requires valid cookies for restricted content

3. **Testing Cookie Functionality**:
   - Use the included debug script: `python test_cookies.py`
   - Try accessing videos directly with yt-dlp:
     ```bash
     yt-dlp --cookies ~/cookies/youtube.txt -F "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
     ```

### WebSocket Connection Issues

If the WebSocket connection fails:

1. Check your CORS settings in the API configuration
2. Verify that the frontend is connecting to the correct WebSocket URL
3. Check for any network issues or proxies that might block WebSocket connections

## Maintenance

### Updating Cookie Files

YouTube cookies typically expire after 2-4 weeks. To maintain access to restricted content:

1. Log in to YouTube in your browser again
2. Export new cookies using the browser extension
3. Replace the existing `youtube.txt` file
4. Restart the application if necessary

### Updating yt-dlp

yt-dlp needs to be updated periodically to work with YouTube:

```bash
pip install -U yt-dlp
```

## Integration Testing

Run the integration test to verify system functionality:

```bash
python integration_test.py
```

This tests:
- WebSocket broadcasting
- Pipeline management
- Error handling
- Multiple pipeline execution

## License

[MIT License](LICENSE)

## Acknowledgements

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube video downloading
- [FFmpeg](https://ffmpeg.org/) for video processing
- [Google Gemini API](https://ai.google.dev/) for advanced vision analysis
- [FastAPI](https://fastapi.tiangolo.com/) for the backend API
- [React](https://reactjs.org/) for the frontend UI