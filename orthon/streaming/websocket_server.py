"""
ORTHON WebSocket Server

FastAPI + WebSocket server for real-time PRISM analysis dashboard.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from .analyzers import RealTimeAnalyzer
from .data_sources import get_stream_connector, DATA_SOURCES

app = FastAPI(title="PRISM Live Analysis Server")

# Serve static dashboard files
dashboard_dir = Path(__file__).parent.parent / "dashboard" / "static"
if dashboard_dir.exists():
    app.mount("/static", StaticFiles(directory=dashboard_dir), name="static")


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, default=str))
            except Exception:
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


@app.get("/")
async def dashboard():
    """Serve the main dashboard."""
    dashboard_path = dashboard_dir / "dashboard.html"
    if dashboard_path.exists():
        with open(dashboard_path) as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html>
        <head><title>PRISM Live Analysis</title></head>
        <body>
            <h1>PRISM Live Analysis Dashboard</h1>
            <p>Dashboard files not found. Please ensure dashboard/static/ directory exists.</p>
            <p>Available endpoints:</p>
            <ul>
                <li><a href="/api/sources">GET /api/sources</a> - List data sources</li>
                <li><a href="/api/status">GET /api/status</a> - Server status</li>
                <li>WebSocket: ws://host:port/ws/{source_type}</li>
            </ul>
        </body>
        </html>
        """)


@app.websocket("/ws/{source_type}")
async def websocket_endpoint(websocket: WebSocket, source_type: str):
    """WebSocket endpoint for live data streaming."""
    await manager.connect(websocket)

    # Initialize analyzer and data source
    try:
        analyzer = RealTimeAnalyzer()
        data_source = get_stream_connector(source_type)
        stream = data_source.stream()

        # Send initial status
        await websocket.send_text(json.dumps({
            'type': 'status',
            'source_type': source_type,
            'message': f'Connected to {source_type} data stream',
            'analyzer_config': analyzer.config
        }))

        # Main analysis loop
        last_update_time = time.time()
        update_count = 0

        while True:
            try:
                # Get new data point (run in executor to not block)
                loop = asyncio.get_event_loop()
                data_point = await loop.run_in_executor(None, lambda: next(stream))

                # Process with PRISM
                instant_results, batch_results = analyzer.process_data_point(data_point)

                # Calculate update rate
                current_time = time.time()
                update_count += 1
                if current_time - last_update_time >= 5.0:  # Every 5 seconds
                    update_rate = update_count / (current_time - last_update_time)
                    last_update_time = current_time
                    update_count = 0
                else:
                    update_rate = None

                # Prepare message
                message = {
                    'type': 'analysis_update',
                    'timestamp': current_time,
                    'instant_results': instant_results,
                    'batch_results': batch_results,
                    'alerts': analyzer.alerts[-5:],  # Last 5 alerts
                    'status_summary': analyzer.get_status_summary(),
                    'raw_data': {k: float(v) for k, v in data_point.items()} if isinstance(data_point, dict) else {},
                }

                if update_rate is not None:
                    message['update_rate'] = update_rate

                # Send to client
                await websocket.send_text(json.dumps(message, default=str))

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)

            except StopIteration:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': 'Data stream ended'
                }))
                break

            except Exception as e:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': f'Analysis error: {str(e)}'
                }))
                await asyncio.sleep(1)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                'type': 'error',
                'message': f'Connection error: {str(e)}'
            }))
        except:
            pass
        manager.disconnect(websocket)


@app.get("/api/sources")
async def list_data_sources():
    """List available data sources."""
    return {
        'sources': list(DATA_SOURCES.keys()),
        'descriptions': {
            'crypto': 'Real-time cryptocurrency market data (Binance)',
            'turbofan': 'Simulated NASA turbofan engine data',
            'reactor': 'Simulated chemical reactor data',
            'system': 'Local system performance metrics',
            'synthetic': 'Synthetic data for testing'
        }
    }


@app.get("/api/status")
async def get_status():
    """Get server status."""
    return {
        'active_connections': len(manager.active_connections),
        'server_time': time.time(),
        'status': 'running'
    }


# Global configuration (set by run_server)
server_config = {
    'default_source': 'turbofan',
    'window_size': 100,
    'batch_size': 500,
}


def run_server(
    source_type: str = "turbofan",
    host: str = "127.0.0.1",
    port: int = 8080,
    window_size: int = 100,
    batch_size: int = 500,
):
    """Run the WebSocket server."""
    import uvicorn

    # Update global config
    server_config['default_source'] = source_type
    server_config['window_size'] = window_size
    server_config['batch_size'] = batch_size

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
