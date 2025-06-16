from fastmcp import Client
import asyncio

async def test_remote_tool():
    # 连接 SSE 服务（URL 必须包含 /sse 路径）
    async with Client("/Users/bytedance/Desktop/python/MCP/ysx/server_stdio.py") as client:  
        result = await client.call_tool("list_directory", arguments={"path":r"/Users/bytedance/Desktop/python/MCP"})
        print("远程调用结果:", result)  # 输出：8

asyncio.run(test_remote_tool())