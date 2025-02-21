import React, { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [analysisResults, setAnalysisResults] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [url, setUrl] = useState('');
  const [duration, setDuration] = useState(30);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeUrl, setActiveUrl] = useState(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/analysis');

    ws.onopen = () => {
      setIsConnected(true);
      setError(null);
      console.log('Connected to WebSocket');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('Received data:', data);
      if (data.type === 'analysis') {
        setAnalysisResults(prev => [...prev, data]);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('WebSocket connection error');
      setIsConnected(false);
    };

    ws.onclose = () => {
      console.log('Disconnected from WebSocket');
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsProcessing(true);
    setAnalysisResults([]); // Clear previous results
    setActiveUrl(url); // Store the active URL
    
    try {
      const response = await fetch('http://localhost:8000/start-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: url,
          chunk_duration: duration,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start analysis');
      }

      await response.json();
    } catch (err) {
      setError(err.message);
      setIsProcessing(false);
      setActiveUrl(null);
    }
  };

  const handleStop = async () => {
    if (!activeUrl) return;
    
    try {
        console.log('Stopping analysis for:', activeUrl);
        const response = await fetch('http://localhost:8000/stop-analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                url: activeUrl
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to stop analysis');
        }

        await response.json();
        console.log('Analysis stopped successfully');
        
        setIsProcessing(false);
        setActiveUrl(null);
        setAnalysisResults([]); // Clear results when stopping
    } catch (err) {
        console.error('Stop error:', err);
        setError(err.message);
    }
  };

  return (
    <div className="dashboard">
      <div className="container">
        <h1>YouTube Stream Analysis Dashboard</h1>
        
        <div className="status-bar">
          Status: {isConnected ? 
            <span className="connected">Connected</span> : 
            <span className="disconnected">Disconnected</span>
          }
        </div>

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        <div className="control-panel">
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="url">YouTube Stream URL:</label>
              <input
                type="text"
                id="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="Enter YouTube URL"
                required
                className="input-field"
                disabled={isProcessing}
              />
            </div>

            <div className="form-group">
              <label htmlFor="duration">Chunk Duration (seconds):</label>
              <input
                type="range"
                id="duration"
                min="10"
                max="300"
                step="10"
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                className="slider"
                disabled={isProcessing}
              />
              <span className="duration-value">{duration} seconds</span>
            </div>

            <div className="button-group">
              <button 
                type="submit" 
                className="submit-button"
                disabled={!isConnected || isProcessing}
              >
                Start Analysis
              </button>

              {isProcessing && (
                <button 
                  type="button" 
                  className="stop-button"
                  onClick={handleStop}
                >
                  Stop Analysis
                </button>
              )}
            </div>
          </form>
        </div>

        <div className="results-section">
          <div className="results-container">
            <h3>Analysis Results</h3>
            <div className="results-list">
              {analysisResults.length === 0 ? (
                isProcessing ? (
                  <div className="loading-container">
                    <Loader2 className="loading-spinner" size={40} />
                    <p>Waiting for results...</p>
                  </div>
                ) : (
                  <p className="no-results">No results yet</p>
                )
              ) : (
                analysisResults.map((result, index) => (
                  <div key={index} className="result-item">
                    <div className="markdown-content">
                      <ReactMarkdown>{result.analysis}</ReactMarkdown>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;