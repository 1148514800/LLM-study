# client_call_tool.py
from openai import OpenAI
from mcp.client import connect_stdio
import asyncio

async def main():
    # 连接到 MCP 工具服务
    client = await connect_stdio("/Users/bytedance/Desktop/python/MCP/ysx/server_stdio.py")

    # 获取 OpenAI 函数工具定义（自动从 MCP 获取 schema）
    tools = await client.list_openai_tools()

    # 模拟 GPT 决定调用某个工具（function_call）
    response = await client.call_tool("list_directory", {"path": "."})
    print("MCP 工具调用结果:", response)

    # # 如果你想让 GPT 决定是否调用工具，可以配合 GPT 接口：
    # # 以下是一个伪代码接口演示（真实调用需要注册 function_tool）
    # openai = OpenAI(api_key="sk-...")  # 你的 API Key
    # chat_response = openai.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "user", "content": "当前目录下有哪些文件？"}
    #     ],
    #     tools=tools,  # MCP 自动转换为 OpenAI tool schema
    #     tool_choice="auto"
    # )
    # print(chat_response)

asyncio.run(main())
