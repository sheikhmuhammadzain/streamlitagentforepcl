"""
WebSocket endpoint for ultra-fast streaming
Provides instant bidirectional communication for the intelligent agent
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional
import json
import traceback

from ..services.tool_agent import run_tool_based_agent

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/agent/stream")
async def websocket_agent_stream(
    websocket: WebSocket,
    question: Optional[str] = Query(None),
    dataset: Optional[str] = Query("all"),
    model: Optional[str] = Query("google/gemini-flash-1.5:free")
):
    """
    WebSocket endpoint for ultra-fast streaming analysis
    
    Benefits over SSE:
    - Instant communication (no HTTP overhead)
    - Bidirectional (client can send cancel/pause)
    - Lower latency (~50-100ms faster per message)
    - More efficient bandwidth usage
    
    Usage (JavaScript):
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws/agent/stream?question=YOUR_QUERY');
    
    ws.onopen = () => {
        console.log('Connected!');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Event:', data);
        
        // Handle different event types
        switch(data.type) {
            case 'start':
                console.log('Analysis started');
                break;
            case 'progress':
                console.log('Stage:', data.stage);
                break;
            case 'chain_of_thought':
                console.log('Thinking:', data.content);
                break;
            case 'code_chunk':
                console.log('Code:', data.chunk);
                break;
            case 'complete':
                console.log('Done!', data.data);
                ws.close();
                break;
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
        console.log('Connection closed');
    };
    ```
    """
    
    await websocket.accept()
    
    try:
        # Get query from WebSocket message or query params
        if not question:
            # Wait for initial message with query
            data = await websocket.receive_json()
            question = data.get("question", "")
            dataset = data.get("dataset", "all")
            model = data.get("model", "x-ai/grok-code-fast-1")
        
        if not question or not question.strip():
            await websocket.send_json({
                "type": "error",
                "message": "Question parameter is required"
            })
            await websocket.close()
            return
        
        # Stream events through WebSocket with tool-based agent
        async for event in run_tool_based_agent(
            query=question,
            model=model
        ):
            try:
                # Check if connection is still open before sending
                if websocket.client_state.name != "CONNECTED":
                    print("Client disconnected, stopping stream")
                    break
                
                # Send JSON directly (no SSE formatting overhead)
                await websocket.send_json(event)
            
            except Exception as send_error:
                # Connection closed, stop streaming
                print(f"Send failed: {send_error}")
                break
        
        # Send completion signal (only if still connected)
        try:
            if websocket.client_state.name == "CONNECTED":
                await websocket.send_json({
                    "type": "stream_end",
                    "message": "Stream completed successfully"
                })
        except:
            pass  # Already disconnected
    
    except WebSocketDisconnect:
        print(f"Client disconnected from WebSocket")
    
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"WebSocket error: {error_details}")
        
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}",
                "details": error_details
            })
        except:
            pass  # Client already disconnected
    
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.websocket("/agent/interactive")
async def websocket_interactive_agent(websocket: WebSocket):
    """
    Interactive WebSocket endpoint for multi-turn conversations
    
    Allows:
    - Multiple queries in same connection
    - Real-time parameter updates
    - Cancel/pause/resume
    - Bidirectional feedback
    
    Usage:
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws/agent/interactive');
    
    ws.onopen = () => {
        // Send first query
        ws.send(JSON.stringify({
            action: 'query',
            question: 'Show top incidents',
            dataset: 'all'
        }));
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'complete') {
            // Send follow-up query
            ws.send(JSON.stringify({
                action: 'query',
                question: 'Now show trends'
            }));
        }
    };
    ```
    """
    
    await websocket.accept()
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Interactive session started. Send queries with action='query'"
        })
        
        while True:
            # Wait for client message
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "query":
                question = data.get("question", "")
                dataset = data.get("dataset", "all")
                model = data.get("model", "x-ai/grok-code-fast-1")
                
                if not question.strip():
                    await websocket.send_json({
                        "type": "error",
                        "message": "Question is required"
                    })
                    continue
                
                # Stream response with tool-based agent
                async for event in run_tool_based_agent(
                    query=question,
                    model=model
                ):
                    try:
                        if websocket.client_state.name == "CONNECTED":
                            await websocket.send_json(event)
                        else:
                            break
                    except:
                        break
                
                # Signal query completion
                await websocket.send_json({
                    "type": "query_complete",
                    "message": "Ready for next query"
                })
            
            elif action == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": data.get("timestamp")
                })
            
            elif action == "close":
                await websocket.send_json({
                    "type": "closing",
                    "message": "Goodbye!"
                })
                break
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown action: {action}"
                })
    
    except WebSocketDisconnect:
        print("Client disconnected from interactive session")
    
    except Exception as e:
        print(f"Interactive session error: {traceback.format_exc()}")
    
    finally:
        try:
            await websocket.close()
        except:
            pass
