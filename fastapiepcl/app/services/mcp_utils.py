"""
MCP (Model Context Protocol) Utilities

This module provides utilities for working with the FastMCP server,
including tool listing, server management, and integration helpers.
"""

from typing import List, Dict, Any
import json


async def get_mcp_server():
    """
    Get the initialized MCP server instance.
    
    Returns:
        FastMCP: The MCP server instance
    """
    from .tool_agent import mcp_server
    return mcp_server


async def list_mcp_tools() -> List[Dict[str, Any]]:
    """
    List all available MCP tools registered with the server.
    
    Returns:
        List of tool definitions with names, descriptions, and parameters
    """
    server = await get_mcp_server()
    tools = server.get_tools()
    
    tool_list = []
    for tool_name, tool in tools.items():
        tool_info = {
            "name": tool_name,
            "description": tool.description if hasattr(tool, 'description') else "No description available",
            "enabled": tool.enabled if hasattr(tool, 'enabled') else True
        }
        tool_list.append(tool_info)
    
    return tool_list


async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """
    Call an MCP tool by name with the provided arguments.
    
    Args:
        tool_name: Name of the MCP tool to call
        arguments: Dictionary of arguments to pass to the tool
    
    Returns:
        The result from the tool execution
    
    Raises:
        ValueError: If the tool is not found
        Exception: If tool execution fails
    """
    server = await get_mcp_server()
    tools = server.get_tools()
    
    if tool_name not in tools:
        available_tools = list(tools.keys())
        raise ValueError(
            f"Tool '{tool_name}' not found. Available tools: {available_tools}"
        )
    
    tool = tools[tool_name]
    
    # Execute the tool
    try:
        result = await tool(**arguments) if hasattr(tool, '__call__') else None
        return result
    except Exception as e:
        raise Exception(f"Failed to execute tool '{tool_name}': {str(e)}")


def get_mcp_tool_schema(tool_name: str = None) -> Dict[str, Any]:
    """
    Get the JSON schema for MCP tools.
    
    Args:
        tool_name: Optional specific tool name. If None, returns all tools.
    
    Returns:
        Dictionary containing tool schema information
    """
    from .tool_agent import TOOLS
    
    if tool_name:
        # Find specific tool
        for tool in TOOLS:
            if tool.get("function", {}).get("name") == tool_name:
                return tool
        return {"error": f"Tool '{tool_name}' not found"}
    
    # Return all MCP tools (those starting with 'mcp_')
    mcp_tools = [
        tool for tool in TOOLS 
        if tool.get("function", {}).get("name", "").startswith("mcp_")
    ]
    
    return {
        "total_mcp_tools": len(mcp_tools),
        "tools": mcp_tools
    }


async def get_mcp_server_info() -> Dict[str, Any]:
    """
    Get information about the MCP server including configuration and stats.
    
    Returns:
        Dictionary with server information
    """
    server = await get_mcp_server()
    tools = server.get_tools()
    
    return {
        "server_name": server.name if hasattr(server, 'name') else "SafetyCopilotMCP",
        "version": server.version if hasattr(server, 'version') else "1.0.0",
        "total_tools": len(tools),
        "tool_names": list(tools.keys()),
        "dependencies": server.dependencies if hasattr(server, 'dependencies') else []
    }


def format_mcp_tool_result(result: Any) -> str:
    """
    Format an MCP tool result for display.
    
    Args:
        result: The result from an MCP tool call
    
    Returns:
        Formatted string representation
    """
    if isinstance(result, str):
        try:
            # Try to parse as JSON for pretty printing
            parsed = json.loads(result)
            return json.dumps(parsed, indent=2)
        except:
            return result
    elif isinstance(result, dict):
        return json.dumps(result, indent=2)
    else:
        return str(result)


# Export convenience functions
__all__ = [
    'get_mcp_server',
    'list_mcp_tools',
    'call_mcp_tool',
    'get_mcp_tool_schema',
    'get_mcp_server_info',
    'format_mcp_tool_result'
]
