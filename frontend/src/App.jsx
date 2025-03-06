import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Loader2, Upload, Info, Play, StopCircle, AlertCircle, RefreshCw, Settings, ChevronDown, ChevronUp, Cpu, Sparkles, Clock } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './App.css';
import PipelineVisualizer from './components/PipelineVisualizer';
import config from './config';

// Custom link renderer that opens links in new tabs
const CustomLink = ({ node, ...props }) => {
  return <a target="_blank" rel="noopener noreferrer" {...props} />;
};

function App() {
  const [analysisResults, setAnalysisResults] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [url, setUrl] = useState('');
  const [duration, setDuration] = useState(30);
  const [runtimeDuration, setRuntimeDuration] = useState(5); // -1 to indefinite
  const [useWebSearch, setUseWebSearch] = useState(true);
  const [customPrompt, setCustomPrompt] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [activePipelines, setActivePipelines] = useState([]);
  const [backendSettings, setBackendSettings] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [showInactivePipelines, setShowInactivePipelines] = useState(false);
  // New state for selected pipeline results filter
  const [selectedResultsPipeline, setSelectedResultsPipeline] = useState("all");

  const websocket = useRef(null);
  const resultsContainerRef = useRef(null);
  const seenMessages = useRef(new Set());

  // Connect to WebSocket and set up event handlers
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket(`${config.wsBaseUrl}/ws/analysis`);
      websocket.current = ws;
      
      ws.onopen = () => {
        console.log('Connected to WebSocket');
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received data:', data);
        
        // Deduplicate messages using timestamp and more content
        if (data.type === 'analysis') {
          // Create a more unique ID from timestamp and content
          const messageId = `${data.timestamp}-${data.chunk_path || ''}-${data.analysis.substring(0, 30)}`;
          
          if (!seenMessages.current.has(messageId)) {
            seenMessages.current.add(messageId);
            setAnalysisResults(prev => [...prev, data]);
            
            // Auto-scroll to the latest result
            if (resultsContainerRef.current) {
              resultsContainerRef.current.scrollTop = resultsContainerRef.current.scrollHeight;
            }
            
            // Limit the size of the seen messages set to prevent memory issues
            if (seenMessages.current.size > 100) {
              // Convert to array, remove first item, convert back to set
              const messagesArray = Array.from(seenMessages.current);
              messagesArray.shift();
              seenMessages.current = new Set(messagesArray);
            }
          } else {
            console.log('Ignoring duplicate message:', messageId);
          }
        } else if (data.type === 'pipeline_status') {
          // Update pipeline status in our list
          updatePipelineStatus(data.pipeline_id, data.status);
        } else if (data.type === 'error') {
          setError(data.message);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Failed to connect to the analysis service');
        setIsConnected(false);
      };

      ws.onclose = () => {
        console.log('Disconnected from WebSocket');
        setIsConnected(false);
        // Try to reconnect after a delay
        setTimeout(connectWebSocket, 3000);
      };

      return ws;
    };

    const ws = connectWebSocket();

    // Cleanup on unmount
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  // Fetch system status periodically
  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        const response = await fetch(`${config.apiBaseUrl}/health`);
        if (response.ok) {
          const data = await response.json();
          setSystemStatus(data);
        }
      } catch (err) {
        console.error('Error fetching system status:', err);
      }
    };

    // Fetch status immediately and then every 10 seconds
    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 10000);

    return () => clearInterval(interval);
  }, []);

  // Fetch backend settings once on load
  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const response = await fetch(`${config.apiBaseUrl}/settings`);
        if (response.ok) {
          const data = await response.json();
          setBackendSettings(data);
          // Initialize duration with backend default
          if (data.pipeline && data.pipeline.default_chunk_duration) {
            setDuration(data.pipeline.default_chunk_duration);
          }
          // Initialize web search setting
          if (data.analysis && data.analysis.use_web_search !== undefined) {
            setUseWebSearch(data.analysis.use_web_search);
          }
          // Initialize runtime duration with backend default
          //if (data.pipeline && data.pipeline.default_runtime_duration !== undefined) {
          //  setRuntimeDuration(data.pipeline.default_runtime_duration);
          //}
        }
      } catch (err) {
        console.error('Error fetching settings:', err);
      }
    };

    fetchSettings();
  }, []);

  // Fetch active pipelines periodically
  useEffect(() => {
    const fetchPipelines = async () => {
      try {
        const response = await fetch(`${config.apiBaseUrl}/pipelines`);
        if (response.ok) {
          const pipelines = await response.json();
          setActivePipelines(pipelines);
        }
      } catch (err) {
        console.error('Error fetching pipelines:', err);
      }
    };

    // Fetch immediately and then every 5 seconds
    fetchPipelines();
    const interval = setInterval(fetchPipelines, 5000);

    return () => clearInterval(interval);
  }, []);

  // Helper to update a specific pipeline's status
  const updatePipelineStatus = useCallback((pipelineId, newStatus) => {
    setActivePipelines(prev => 
      prev.map(pipeline => 
        pipeline.pipeline_id === pipelineId 
          ? { ...pipeline, ...newStatus } 
          : pipeline
      )
    );
  }, []);

  // Filter pipelines based on active state
  const getFilteredPipelines = () => {
    if (showInactivePipelines) {
      return activePipelines;
    } else {
      return activePipelines.filter(pipeline => 
        !['stopped', 'failed', 'completed'].includes(pipeline.state)
      );
    }
  };

  // Filter results based on selected pipeline
  const getFilteredResults = useCallback(() => {
    if (selectedResultsPipeline === "all") {
      return analysisResults;
    }
    
    // Find the selected pipeline to get its output directory
    const selectedPipeline = activePipelines.find(p => p.pipeline_id === selectedResultsPipeline);
    if (!selectedPipeline) {
      return analysisResults; // Fallback to all results if pipeline not found
    }
    
    // Filter results that contain the output directory path in their chunk_path
    return analysisResults.filter(result => 
      result.chunk_path && result.chunk_path.includes(selectedPipeline.output_dir)
    );
  }, [analysisResults, selectedResultsPipeline, activePipelines]);

  // Start analysis with the new API
  const handleStartAnalysis = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    try {
      // Alternative approach - using the legacy endpoint for better compatibility
      const startResponse = await fetch(`${config.apiBaseUrl}/start-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: url,
          chunk_duration: duration,
          export_responses: true,
          runtime_duration: runtimeDuration // Add runtime duration
        }),
      });

      if (!startResponse.ok) {
        const errorData = await startResponse.json();
        throw new Error(errorData.detail || 'Failed to start analysis');
      }

      const responseData = await startResponse.json();
      console.log('Analysis started with legacy endpoint:', responseData);
      
      // Clear previous results when starting a new analysis
      setAnalysisResults([]);
      
    } catch (legacyErr) {
      console.log('Legacy endpoint failed, trying new API flow...');
      
      try {
        // 1. Register the source
        const sourceResponse = await fetch(`${config.apiBaseUrl}/sources`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            url: url,
            source_type: "youtube",
            metadata: { 
              added_from: "frontend",
              timestamp: new Date().toISOString()
            }
          }),
        });

        if (!sourceResponse.ok) {
          const errorData = await sourceResponse.json();
          throw new Error(errorData.detail || 'Failed to register source');
        }

        const sourceData = await sourceResponse.json();
        const sourceId = sourceData.source.source_id;
        console.log('Source registered with ID:', sourceId);

        // Wait a short time to ensure source is fully registered
        await new Promise(resolve => setTimeout(resolve, 500));

        // 2. Create a pipeline
        const pipelineResponse = await fetch(`${config.apiBaseUrl}/pipelines`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            source_id: sourceId,
            chunk_duration: duration,
            analysis_prompt: customPrompt || undefined,
            use_web_search: useWebSearch,
            runtime_duration: runtimeDuration // Add runtime duration
          }),
        });

        if (!pipelineResponse.ok) {
          const errorData = await pipelineResponse.json();
          throw new Error(errorData.detail || 'Failed to create pipeline');
        }

        const pipelineData = await pipelineResponse.json();
        const pipelineId = pipelineData.pipeline_id;
        console.log('Pipeline created with ID:', pipelineId);

        // 3. Start the pipeline
        const startResponse = await fetch(`${config.apiBaseUrl}/pipelines/${pipelineId}/start`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          }
        });

        if (!startResponse.ok) {
          const errorData = await startResponse.json();
          throw new Error(errorData.detail || 'Failed to start pipeline');
        }

        // Clear previous results when starting a new analysis
        setAnalysisResults([]);
        console.log('Analysis started successfully with pipeline ID:', pipelineId);
        
      } catch (err) {
        console.error('Error starting analysis with new API:', err);
        setError(err.message);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Stop a specific pipeline with enhanced reliability
  const handleStopPipeline = async (pipelineId) => {
    try {
      // Set pipeline to visually "stopping" status immediately for user feedback
      updatePipelineStatus(pipelineId, { state: "stopping" });
      
      // Get the pipeline url before we try stopping it
      const pipeline = activePipelines.find(p => p.pipeline_id === pipelineId);
      const pipelineUrl = pipeline?.url;
      
      // Try the dedicated pipeline stop endpoint
      console.log(`Stopping pipeline ${pipelineId} via pipeline endpoint`);
      const response = await fetch(`${config.apiBaseUrl}/pipelines/${pipelineId}/stop`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.warn(`Primary stop method failed: ${errorData.detail || 'Unknown error'}`);
        
        // If we have the URL, try the legacy stop endpoint as fallback
        if (pipelineUrl) {
          console.log(`Trying legacy stop endpoint for URL: ${pipelineUrl}`);
          const legacyResponse = await fetch(`${config.apiBaseUrl}/stop-analysis`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: pipelineUrl })
          });
          
          if (!legacyResponse.ok) {
            throw new Error('Both stop methods failed');
          }
          console.log('Pipeline stopped via legacy endpoint');
        } else {
          throw new Error('Primary stop failed and URL not available for fallback');
        }
      } else {
        console.log('Pipeline stopped via primary endpoint:', pipelineId);
      }
      
      // Notify user about potential continued streaming
      setError("Pipeline stop requested. Note: Video chunking might continue in the background due to a backend limitation.");
      
      // Set a timeout to clear this message
      setTimeout(() => {
        setError(null);
      }, 10000);
      
    } catch (err) {
      console.error('Error stopping pipeline:', err);
      setError(`Failed to stop pipeline. Error: ${err.message}. The backend may still be processing data.`);
    }
  };

  // Format relative time
  const formatRelativeTime = (timestamp) => {
    if (!timestamp) return 'N/A';
    
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffSec = Math.floor(diffMs / 1000);
    
    if (diffSec < 60) return `${diffSec} seconds ago`;
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)} minutes ago`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)} hours ago`;
    return `${Math.floor(diffSec / 86400)} days ago`;
  };

  // Get state color for visual indicators
  const getStateColor = (state) => {
    const stateColors = {
      'running': 'bg-green-500',
      'starting': 'bg-blue-500',
      'stopping': 'bg-orange-500',
      'stopped': 'bg-gray-500',
      'failed': 'bg-red-500',
      'completed': 'bg-purple-500',
      'paused': 'bg-yellow-500',
      'pausing': 'bg-yellow-300',
      'initialized': 'bg-gray-300'
    };
    return stateColors[state] || 'bg-gray-500';
  };

  // Format runtime duration for display
  const formatRuntimeDuration = (minutes) => {
    if (minutes === -1) return "Indefinite";
    if (minutes < 60) return `${minutes} minutes`;
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    if (remainingMinutes === 0) return `${hours} hour${hours > 1 ? 's' : ''}`;
    return `${hours} hour${hours > 1 ? 's' : ''} ${remainingMinutes} minute${remainingMinutes > 1 ? 's' : ''}`;
  };

  const filteredPipelines = getFilteredPipelines();

  return (
    <div className="dashboard">
      <div className="container">
        <h1>YouTube Stream Analysis Dashboard</h1>
        
        <div className="status-panel">
          <div className="status-bar">
            Connection: {isConnected ? 
              <span className="connected">Connected</span> : 
              <span className="disconnected">Disconnected</span>
            }
            {systemStatus && (
              <div className="system-info">
                <span className="system-status">
                  <Cpu size={16} className="icon" />
                  Active Pipelines: {systemStatus.active_pipelines}/{systemStatus.total_pipelines}
                </span>
                <span className="system-version">
                  <span>v{systemStatus.version}</span>
                </span>
              </div>
            )}
          </div>

          {error && (
            <div className="error">
              <AlertCircle size={18} />
              {error}
            </div>
          )}
        </div>

        <div className="panel-container">
          <div className="left-panel">
            <div className="control-panel">
              <h2><Upload size={20} className="panel-icon" /> Start New Analysis</h2>
              <form onSubmit={handleStartAnalysis}>
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
                    disabled={isLoading}
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="duration">
                    Chunk Duration: <span className="duration-value">{duration} seconds</span>
                  </label>
                  <input
                    type="range"
                    id="duration"
                    min={backendSettings?.pipeline?.min_chunk_duration || 10}
                    max={backendSettings?.pipeline?.max_chunk_duration || 300}
                    step="10"
                    value={duration}
                    onChange={(e) => setDuration(Number(e.target.value))}
                    className="slider"
                    disabled={isLoading}
                  />
                </div>

                <div className="advanced-toggle" onClick={() => setShowAdvanced(!showAdvanced)}>
                  <Settings size={16} className="icon" />
                  Advanced Options
                  {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </div>

                {showAdvanced && (
                  <div className="advanced-options">
                    <div className="form-group checkbox-group">
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={useWebSearch}
                          onChange={(e) => setUseWebSearch(e.target.checked)}
                          disabled={isLoading}
                        />
                        <Sparkles size={16} className="icon" />
                        Use web search for enhanced analysis
                      </label>
                    </div>

                    <div className="form-group">
                      <label htmlFor="runtimeDuration" className="flex items-center gap-2">
                        <Clock size={16} className="icon" />
                        Runtime Duration:
                      </label>
                      <select
                        id="runtimeDuration"
                        value={runtimeDuration}
                        onChange={(e) => setRuntimeDuration(Number(e.target.value))}
                        className="input-field"
                        disabled={isLoading}
                      >
                        <option value="-1">Run indefinitely (until manually stopped)</option>
                        <option value="5">5 minutes</option>
                        <option value="10">10 minutes</option>
                        <option value="15">15 minutes</option>
                        <option value="30">30 minutes</option>
                        <option value="60">1 hour</option>
                        <option value="120">2 hours</option>
                        <option value="180">3 hours</option>
                        <option value="240">4 hours</option>
                      </select>
                      <div className="text-sm mt-1 text-gray-600">
                        {runtimeDuration > 0 && `Pipeline will automatically stop after ${formatRuntimeDuration(runtimeDuration)}`}
                        {runtimeDuration === -1 && "Pipeline will run until manually stopped"}
                      </div>
                    </div>

                    <div className="form-group">
                      <label htmlFor="customPrompt">Custom Analysis Prompt:</label>
                      <textarea
                        id="customPrompt"
                        value={customPrompt}
                        onChange={(e) => setCustomPrompt(e.target.value)}
                        placeholder="Leave empty for default prompt"
                        className="textarea-field"
                        disabled={isLoading}
                        rows={3}
                      />
                    </div>
                  </div>
                )}

                <div className="button-group">
                  <button 
                    type="submit" 
                    className="submit-button"
                    disabled={!isConnected || isLoading || !url.trim()}
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="spin-icon" size={18} />
                        Starting...
                      </>
                    ) : (
                      <>
                        <Play size={18} />
                        Start Analysis
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>

            {/* Pipeline Visualizer */}
            {filteredPipelines.length > 0 && (
              <PipelineVisualizer pipelines={filteredPipelines} />
            )}

            <div className="pipeline-list">
              <div className="pipeline-header-row">
                <h2><Cpu size={20} className="panel-icon" /> Active Pipelines</h2>
                {activePipelines.length > 0 && (
                  <button 
                    className="pipeline-toggle-button"
                    onClick={() => setShowInactivePipelines(!showInactivePipelines)}
                  >
                    {showInactivePipelines ? "Hide Inactive" : "Show All"}
                  </button>
                )}
              </div>
              
              {filteredPipelines.length === 0 ? (
                <div className="no-pipelines">
                  <p>No active pipelines</p>
                </div>
              ) : (
                <div className="pipelines">
                  {filteredPipelines.map(pipeline => (
                    <div key={pipeline.pipeline_id} className={`pipeline-item ${pipeline.state === 'stopping' ? 'stopping' : ''}`}>
                      <div className="pipeline-header">
                        <div className="pipeline-state">
                          <span className={`state-indicator ${pipeline.state}`}></span>
                          <span className="state-text">{pipeline.state}</span>
                          {pipeline.state === 'stopping' && (
                            <span className="stopping-note">(chunking may continue)</span>
                          )}
                        </div>
                        <div className="pipeline-actions">
                          {(pipeline.state === 'running' || pipeline.state === 'starting') && (
                            <button 
                              className="stop-pipeline-button"
                              onClick={() => handleStopPipeline(pipeline.pipeline_id)}
                            >
                              <StopCircle size={16} />
                              Stop
                            </button>
                          )}
                        </div>
                      </div>
                      
                      <div className="pipeline-details">
                        <div className="pipeline-url" title={pipeline.url}>
                          {pipeline.url}
                        </div>
                        <div className="pipeline-stats">
                          <span>
                            <strong>Chunks:</strong> {pipeline.stats.chunks_processed}
                          </span>
                          <span>
                            <strong>Created:</strong> {formatRelativeTime(pipeline.created_at)}
                          </span>
                          {pipeline.stats.errors > 0 && (
                            <span className="pipeline-errors">
                              <strong>Errors:</strong> {pipeline.stats.errors}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="right-panel">
            <div className="results-section">
              <div className="results-header">
                <h2>
                  <Info size={20} className="panel-icon" /> 
                  Analysis Results
                  {selectedResultsPipeline !== "all" && (
                    <span className="filtered-count">
                      ({getFilteredResults().length}/{analysisResults.length})
                    </span>
                  )}
                </h2>
                <div className="results-controls">
                  <select 
                    className="pipeline-filter-select"
                    value={selectedResultsPipeline}
                    onChange={(e) => setSelectedResultsPipeline(e.target.value)}
                  >
                    <option value="all">All Pipelines</option>
                    {filteredPipelines.map(pipeline => (
                    <option key={pipeline.pipeline_id} value={pipeline.pipeline_id} title={pipeline.url}>
                      {pipeline.url}
                    </option>
                    ))}
                  </select>
                  {analysisResults.length > 0 && (
                    <button 
                      className="clear-results-button"
                      onClick={() => setAnalysisResults([])}
                    >
                      <RefreshCw size={16} />
                      Clear
                    </button>
                  )}
                </div>
              </div>
              
              <div className="results-container" ref={resultsContainerRef}>
                {analysisResults.length === 0 ? (
                  <div className="no-results">
                    <p>No results yet</p>
                    <p className="instruction">Start an analysis to see results here</p>
                  </div>
                ) : (
                  <div className="results-list">
                    {getFilteredResults().map((result, index) => (
                      <div key={index} className="result-item">
                        <div className="result-timestamp">
                          {new Date(result.timestamp).toLocaleTimeString()}
                        </div>
                        <div className="markdown-content">
                          {/* Use custom link component to open links in new tabs */}
                          <ReactMarkdown components={{ a: CustomLink }}>
                            {result.analysis}
                          </ReactMarkdown>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;