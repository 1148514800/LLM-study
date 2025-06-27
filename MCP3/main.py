'''
将工具集成到一个json文件中，通过json文件进行调用。
'''

import asyncio
from fastmcp.client.transports import NpxStdioTransport,UvxStdioTransport,PythonStdioTransport

from core import agent
from interface import run
from utils import get_all_tools

# from tools import add, mul, compare, count_letter_in_string


async def main():

    # 获取MCP工具和本地工具
    MCP_tools, local_tools = get_all_tools()

    agent_run = agent(MCP_tools=MCP_tools, local_tools=local_tools, model="Qwen/Qwen2.5-72B-Instruct", tool_show=True)
    await run(agent_run)

if __name__ == '__main__':
    asyncio.run(main())
