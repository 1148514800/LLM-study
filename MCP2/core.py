# agent.py
import asyncio
import json
from typing import List, Dict

from fastmcp import Client
from openai import OpenAI
from fastmcp.client.transports import SSETransport, StreamableHttpTransport, NpxStdioTransport, UvxStdioTransport, PythonStdioTransport

# from tools import add, mul, compare, count_letter_in_string
from utils import function_to_json


class agent:

    def __init__(self, MCP_tools: list = None, local_tools: dict = None,model="Qwen/Qwen2.5-14B-Instruct", tool_show=False):
        self.model = model
        self.mcp_clients = [Client(MCP_tool) for MCP_tool in MCP_tools]
        self.local_tools = local_tools
        self.tool_show = tool_show
        self.openai_client = OpenAI(
            api_key="sk-jthfgsvomtshbuvlswlbajmnxzwfxvvhbmlfqoggdsvreuez",  # 使用安全方式加载建议放在 env 里
            base_url="https://api.siliconflow.cn/v1",
        )
        # self.openai_client = OpenAI(
        #     api_key="sk-Lq3vLuVpw1fTfYUVeo6w5j3MIfRzl0pgHySNjYiPUnuDZxYE",  # 使用安全方式加载建议放在 env 里
        #     base_url="https://api.gptgod.online/v1/",
        # )
        self.messages = [{
            "role": "system",
            "content": "你是一个有用的助手，如果用户的问题需要调用工具，那你使用帮助用户回答问题；不需要调用工具则直接输出结果"
        }]
        self.tools = []
        self.MCP_tools = {}
        # self.local_tools = self._prepare_local_tool_funcs()

    # def _prepare_local_tool_funcs(self):
    #     return {
    #         "add": add,
    #         "mul": mul,
    #         "compare": compare,
    #         "count_letter_in_string": count_letter_in_string
    #     }

    def _get_local_tool_schemas(self):
        return [function_to_json(func) for func in self.local_tools.values()]

    async def prepare_tools(self):
        all_tools = self._get_local_tool_schemas()

        for client in self.mcp_clients:
            tools = await client.list_tools()
            for tool in tools:
                tool_spec = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    }
                }
                all_tools.append(tool_spec)
                self.MCP_tools[tool.name] = client

        return all_tools

    async def chat(self, messages: List[Dict]):
        if not self.tools:
            self.tools = await self.prepare_tools()

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools
        )

        if response.choices[0].finish_reason != "tool_calls":
            return response.choices[0].message

        self.messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })

        for tool_call in response.choices[0].message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if self.tool_show:
                print(f"Tool call: {tool_name} with arguments: {arguments}")

            if tool_name in self.MCP_tools:
                client = self.MCP_tools[tool_name]
                tool_response = await client.call_tool(tool_name, arguments)
                content = tool_response[0].text
            elif tool_name in self.local_tools:
                content = self.local_tools[tool_name](**arguments)
            else:
                content = f"Unknown tool: {tool_name}"

            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": content
            })

        return await self.chat(self.messages)

    async def enter_clients(self):
        self.entered_clients = []
        for client in self.mcp_clients:
            entered = await client.__aenter__()
            self.entered_clients.append(entered)

    async def exit_clients(self):
        for client in self.entered_clients:
            await client.__aexit__(None, None, None)
