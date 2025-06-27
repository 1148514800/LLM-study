'''
1、创建客户端
2、获取工具，（资源、prompt）
3、调用工具
'''

from fastmcp import Client
import asyncio
from fastmcp.client.transports import SSETransport,StreamableHttpTransport,NpxStdioTransport,UvxStdioTransport,PythonStdioTransport

async def run():
    transport = NpxStdioTransport(package="howtocook-mcp", args=["-y"])
    client = Client(transport)
    async with client:
        # 获取工具列表
        tools = await client.list_tools()   
        print("可用工具列表:", tools)

        # 调用工具
        tool = tools[-2]
        tool_result = await client.call_tool(
            tool.name, 
            arguments={"people": 3}
        )
        print(f"调用工具 {tool.name} 的结果:", tool_result)

if __name__ == "__main__":
    asyncio.run(run())