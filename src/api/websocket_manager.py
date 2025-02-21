from typing import List
from fastapi import WebSocket
from datetime import datetime

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast_analysis(self, result: str, chunk_path: str):
        message = {
            "type": "analysis",
            "chunk_path": chunk_path,
            "analysis": result,
            "timestamp": str(datetime.now())
        }
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except:
                self.active_connections.remove(connection)

# Create a singleton instance
manager = WebSocketManager()