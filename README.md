# Live-Vision Deployment Guide

This guide explains how to deploy the Live-Vision project on a Linux server from scratch. The project consists of a FastAPI backend and a React frontend for real-time YouTube stream analysis using Google's Gemini Vision API.

## Project Structure

The deployed project should have this structure:
```
/opt/live-vision/
├── env/                  # Python virtual environment
├── src/                  # Backend source code
│   ├── analyzer/        # Gemini Vision analysis
│   ├── api/            # FastAPI endpoints
│   ├── pipeline/       # Video processing pipeline
│   └── recorder/       # YouTube stream handling
├── frontend/            
│   ├── dist/            # Built frontend files
│   └── node_modules/    # Frontend dependencies
├── data/                # Video chunks directory
├── .env                 # Environment variables
└── requirements.txt     # Python dependencies
```

## Prerequisites

### System Requirements
- Linux server (Ubuntu 20.04 or later recommended)
- Python 3.9 or later
- Node.js 18 or later
- npm 9 or later
- FFmpeg
- yt-dlp
- Google Gemini API key

## Deployment Steps

### 1. Initial Server Setup

```bash
sudo -i

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
sudo apt install python3-pip python3-venv -y

# Install Node.js and npm
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs -y

# Install FFmpeg
sudo apt install ffmpeg -y

# Install yt-dlp
sudo curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp
sudo chmod a+rx /usr/local/bin/yt-dlp

### 2. Clone and Set Up Project

```bash
# Clone the repository
git clone https://github.com/Reynold97/Live-Vision.git

# Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Setup frontend
cd frontend
npm install
npm install lucide-react react-markdown
npm run build
cd ..
```

### 3. Environment Setup

Create a `.env` file in the project root:

```bash
# Create and edit .env file
nano .env
```

Add the following content:
```env
GOOGLE_API_KEY=your_gemini_api_key
```

### 4. Setup Systemd Service for Backend

Create a service file for the backend:

```bash
sudo nano /etc/systemd/system/live-vision.service
```

Add the following content:
```ini
[Unit]
Description=Live-Vision Backend Service
After=network.target

[Service]
User=your_user
Group=your_group
WorkingDirectory=/root/Live-Vision
Environment="PATH=/root/Live-Vision/env/bin"
EnvironmentFile=/root/Live-Vision/.env
ExecStart=/root/Live-Vision/env/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000

[Install]
WantedBy=multi-user.target
```

### 5. Setup Nginx

```bash
# Install Nginx
sudo apt install nginx -y

# Create Nginx configuration
sudo nano /etc/nginx/sites-available/live-vision
```

Add the following configuration:
```nginx
server {
    listen 80;
    server_name http://videoanalyzer.storyface.ai;

    # Frontend
    location / {
        root /root/Live-Vision/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket
    location /ws/ {
        proxy_pass http://localhost:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/live-vision /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6. Start Services

```bash
# Start and enable backend service
sudo systemctl start live-vision
sudo systemctl enable live-vision

# Verify status
sudo systemctl status live-vision
```

## Maintenance

### Updating the Application

```bash
# Stop the service
sudo systemctl stop live-vision

# Pull new changes
git pull

# Update backend
source env/bin/activate
pip install -r requirements.txt

# Update frontend
cd frontend
npm install
npm run build
cd ..

# Restart service
sudo systemctl start live-vision
```

### Logs

View backend logs:
```bash
sudo journalctl -u live-vision -f
```

### Common Issues

1. Permission Issues:
```bash
# Fix data directory permissions
sudo chown -R your_user:your_group /opt/live-vision/data
sudo chmod 755 /opt/live-vision/data
```

2. WebSocket Connection Issues:
- Check Nginx configuration
- Verify firewall settings: `sudo ufw status`

3. Video Processing Issues:
- Verify FFmpeg installation: `ffmpeg -version`
- Check yt-dlp: `yt-dlp --version`

## Security Setup

### 1. SSL/TLS Configuration

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your_domain.com
```

### 2. Firewall Configuration

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

## Production Considerations

Before deploying to production:

1. Environment Configuration:
   - Update frontend environment variables
   - Set DEBUG=False in backend
   - Configure proper logging

2. Security Measures:
   - Enable SSL/TLS
   - Set up proper firewall rules
   - Configure rate limiting
   - Implement authentication if needed

3. Performance Optimization:
   - Configure Nginx caching
   - Optimize video chunk size
   - Monitor system resources

## Requirements Files

### Backend (requirements.txt)
```
fastapi
uvicorn[standard]
python-dotenv
google-generativeai
watchdog
python-multipart
aiohttp
```

### Frontend (.gitignore)
```gitignore
# Dependencies
/node_modules
/.pnp
.pnp.js

# Production
/dist
/build

# Environment
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Logs
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Editor directories and files
.vscode/*
!.vscode/extensions.json
.idea
.DS_Store
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?
```

## Troubleshooting

1. Frontend not loading:
   - Check Nginx configuration
   - Verify build files in dist directory
   - Check console for errors

2. Backend connection issues:
   - Verify service status
   - Check ports and firewall
   - Verify WebSocket configuration

3. Video processing issues:
   - Check FFmpeg installation
   - Verify yt-dlp is working
   - Check disk space for chunks

