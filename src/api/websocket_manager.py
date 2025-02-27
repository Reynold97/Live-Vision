# src/api/websocket_manager.py
from typing import List, Dict, Any, Optional
from fastapi import WebSocket
from datetime import datetime
import logging
import json
import asyncio

class WebSocketManager:
    """Enhanced WebSocket manager for multi-pipeline system."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        
    async def connect(self, websocket: WebSocket):
        """Connect a new WebSocket client."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
            self.logger.info(f"New WebSocket connection established. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client."""
        try:
            self.active_connections.remove(websocket)
            self.logger.info(f"WebSocket connection closed. Remaining connections: {len(self.active_connections)}")
        except ValueError:
            self.logger.warning("Attempted to disconnect a WebSocket that wasn't connected")

    async def broadcast_analysis(self, result: str, chunk_path: str):
        """Broadcast video analysis results to all connected clients."""
        message = {
            "type": "analysis",
            "chunk_path": chunk_path,
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast_message(message)

    async def broadcast_pipeline_status(self, pipeline_id: str, status: Dict[str, Any]):
        """Broadcast pipeline status updates to all connected clients."""
        message = {
            "type": "pipeline_status",
            "pipeline_id": pipeline_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast_message(message)
        
    async def broadcast_error(self, error_message: str, context: Optional[Dict[str, Any]] = None):
        """Broadcast error messages to all connected clients."""
        message = {
            "type": "error",
            "message": error_message,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast_message(message)
        
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a generic message to all connected clients."""
        if not self.active_connections:
            self.logger.debug("No active connections to broadcast to")
            return
            
        failed_connections = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.warning(f"Failed to send message to client: {str(e)}")
                failed_connections.append(connection)
        
        # Clean up failed connections
        if failed_connections:
            async with self._lock:
                for conn in failed_connections:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)
                        
            self.logger.info(f"Removed {len(failed_connections)} failed connections. Remaining: {len(self.active_connections)}")

# Create a singleton instance
manager = WebSocketManager()