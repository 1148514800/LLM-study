# main.py

import asyncio
from fastmcp.client.transports import NpxStdioTransport, PythonStdioTransport

from core import agent
from interface import run

from tools import add, mul, compare, count_letter_in_string

async def main():
    MCP_tools = [
        NpxStdioTransport(package="mcp-time-server", args=["-y"]),
        PythonStdioTransport("server.py"),
        NpxStdioTransport(package="howtocook-mcp", args=["-y"]),
    ]
    local_tools = {
        "add": add,
        "mul": mul,
        "compare": compare,
        "count_letter_in_string": count_letter_in_string
    }
    agent_run = agent(MCP_tools=MCP_tools, local_tools=local_tools, model="Qwen/Qwen2.5-72B-Instruct", tool_show=True)
    await run(agent_run)

if __name__ == '__main__':
    asyncio.run(main())
