import asyncio
from mcp.client import Client
from mcp.client.stdio import connect_stdio

async def main():
    async with connect_stdio() as transport:
        client = Client(transport)

        # 列出所有可用的工具和资源
        tools = await client.list_tools()
        resources = await client.list_resources()

        print("Available Tools:", tools)
        print("Available Resources:", resources)

        # 调用 add 工具
        result = await client.call_tool("add", {"a": 10, "b": 20})
        print("Add Result:", result.content)  # 应输出 30

        # 获取 greeting 资源
        response = await client.get_resource("greeting://Bob")
        print("Greeting:", response.text)  # 应输出 "Hello, Bob!"

asyncio.run(main())