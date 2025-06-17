from fastmcp import Client
import asyncio
from fastmcp.client.transports import SSETransport,StreamableHttpTransport,NpxStdioTransport,UvxStdioTransport,PythonStdioTransport



async def test_remote_tool():


    transport = PythonStdioTransport(
        script_path="/Users/bytedance/Desktop/python/MCP/ysx/server_stdio.py"
    )

    async with Client(transport) as client:  
        # result = await client.call_tool("list_directory", arguments={"path":r"C:\Users\24853\Desktop"})
        result = await client.call_tool("list_directory", arguments={"path":r"/Users/bytedance/Desktop/python/MCP"})
        print("远程调用结果:", result) 
        await transport.close()## npx和uvx需要
        # 给子进程结束时间
        await asyncio.sleep(0.2)## npx和uvx需要
asyncio.run(test_remote_tool())