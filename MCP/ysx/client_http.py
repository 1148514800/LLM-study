from fastmcp import Client
import asyncio
from fastmcp.client.transports import StreamableHttpTransport
async def test_remote_tool():
    # 连接 SSE 服务（URL 必须包含 /sse 路径）
    # async with Client("file.py") as client:  
    async with Client(StreamableHttpTransport("http://127.0.0.1:8001/mcp")) as client:  
        result = await client.call_tool("list_directory", arguments={"path":r"/Users/bytedance/Desktop/python/MCP"})
        print("远程调用结果:", result)  

asyncio.run(test_remote_tool())