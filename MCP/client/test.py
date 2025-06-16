from openai import OpenAI
import requests
import json

# 初始化 OpenAI 客户端（新版 SDK）
client = OpenAI(
    api_key="sk-jthfgsvomtshbuvlswlbajmnxzwfxvvhbmlfqoggdsvreuez",
    base_url="https://api.siliconflow.cn/v1",
)

MCP_SERVER_BASE = "http://localhost:8003"  # MCP Server 地址

functions = [
    # {
    #     "name": "add",
    #     "description": "Add two numbers",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "a": {"type": "integer", "description": "First number"},
    #             "b": {"type": "integer", "description": "Second number"},
    #         },
    #         "required": ["a", "b"],
    #     },
    # },
    # {
    #     "name": "get_greeting",
    #     "description": "Get a personalized greeting",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "name": {"type": "string", "description": "Name of the person"},
    #         },
    #         "required": ["name"],
    #     },
    # }
    {
        "type": "function",
        "function": {
            "name": "add",            # 函数的名称
            "description": "Add two numbers", # 函数的文档字符串（如果不存在则为空字符串）
            "parameters": {
                "type": "object",
                    "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },     # 函数参数的类型描述
                "required": ["a", "b"],         # 必须参数的列表
            },
        },
    }
]

def call_mcp_tool(tool_name, arguments):
    if tool_name == "add":
        resp = requests.post(f"{MCP_SERVER_BASE}", json=arguments)
    elif tool_name == "get_greeting":
        name = arguments["name"]
        resp = requests.get(f"{MCP_SERVER_BASE}/resource/greeting://{name}")
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

    if resp.ok:
        return resp.json() if "application/json" in resp.headers.get("content-type", "") else resp.text
    else:
        return f"Error calling MCP tool {tool_name}: {resp.status_code}"

def chat_with_tools(prompt):
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-14B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        tools=functions,
        function_call="auto",
    )

    message = response.choices[0].message

    if message.tool_calls:
        name = message.tool_calls[0].function.name
        arguments = json.loads(message.tool_calls[0].function.arguments)
        print(f"[大模型调用工具] name={name}, args={arguments}")

        tool_response = call_mcp_tool(name, arguments)
        print(tool_response)
        followup = client.chat.completions.create(
            model="Qwen/Qwen2.5-14B-Instruct",
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "function_call": message.function_call.model_dump()},
                {"role": "function", "name": name, "content": str(tool_response)}
            ]
        )
        return followup.choices[0].message.content
    else:
        return message.content

if __name__ == "__main__":
    prompt = "帮我计算 1243541512341 + 214413432 = ?"
    result = chat_with_tools(prompt)
    print("🤖 Answer:", result)
