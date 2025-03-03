import React, { useState, useEffect } from 'react';
import { Play, StopCircle, Pause, CheckCircle, Clock, XCircle } from 'lucide-react';

const PipelineVisualizer = ({ pipelines = [] }) => {
  const [selectedPipeline, setSelectedPipeline] = useState(null);
  
  useEffect(() => {
    // Select the first active pipeline by default
    if (pipelines.length > 0) {
      // Prioritize running or starting pipelines
      const activePipeline = pipelines.find(p => 
        p.state === 'running' || p.state === 'starting'
      );
      
      if (activePipeline) {
        setSelectedPipeline(activePipeline.pipeline_id);
      } else if (!selectedPipeline || !pipelines.find(p => p.pipeline_id === selectedPipeline)) {
        // If no active pipeline or selected pipeline no longer exists, select the first one
        setSelectedPipeline(pipelines[0].pipeline_id);
      }
    } else {
      setSelectedPipeline(null);
    }
  }, [pipelines, selectedPipeline]);

  if (pipelines.length === 0) {
    return (
      <div className="pipeline-visualizer">
        <div className="visualizer-empty">
          <p>No pipelines available to visualize</p>
        </div>
      </div>
    );
  }

  // Get the selected pipeline
  const pipeline = pipelines.find(p => p.pipeline_id === selectedPipeline) || pipelines[0];
  
  // Get state icon
  const getStateIcon = (state) => {
    switch (state) {
      case 'running': return <Play className="state-icon running" />;
      case 'starting': return <Clock className="state-icon starting" />;
      case 'stopping': return <Pause className="state-icon stopping" />;
      case 'stopped': return <StopCircle className="state-icon stopped" />;
      case 'failed': return <XCircle className="state-icon failed" />;
      case 'completed': return <CheckCircle className="state-icon completed" />;
      default: return <Clock className="state-icon" />;
    }
  };

  return (
    <div className="pipeline-visualizer">
      {pipelines.length > 1 && (
        <div className="pipeline-selector">
          <label>Select Pipeline:</label>
          <select 
            value={selectedPipeline || ''}
            onChange={(e) => setSelectedPipeline(e.target.value)}
          >
            {pipelines.map(p => (
              <option key={p.pipeline_id} value={p.pipeline_id}>
                {p.url.substring(0, 40)}{p.url.length > 40 ? '...' : ''} - {p.state}
              </option>
            ))}
          </select>
        </div>
      )}
      
      <div className="pipeline-current-state">
        <div className="state-indicator-container">
          <div className={`state-dot ${pipeline.state}`}></div>
          <h3>{pipeline.state}</h3>
        </div>
        <div className="pipeline-stats">
          Processed: <span>{pipeline.stats.chunks_processed} chunks</span>
        </div>
      </div>

      <div className="state-diagram">
        <div className="pipeline-state-description">
          <div className="state-info">
            <div className="state-icon-container">
              {getStateIcon(pipeline.state)}
            </div>
            <div className="state-details">
              <p><strong>Current State:</strong> {pipeline.state}</p>
              <p><strong>Pipeline ID:</strong> {pipeline.pipeline_id.substring(0, 8)}...</p>
              <p><strong>Chunks Processed:</strong> {pipeline.stats.chunks_processed}</p>
              {pipeline.stats.errors > 0 && (
                <p className="error-count"><strong>Errors:</strong> {pipeline.stats.errors}</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PipelineVisualizer;