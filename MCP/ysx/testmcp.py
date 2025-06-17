from fastmcp import Client
import asyncio
from fastmcp.client.transports import SSETransport,StreamableHttpTransport,NpxStdioTransport,UvxStdioTransport,PythonStdioTransport



async def test_remote_tool():

    # # 1、Npx调用
    # transport = NpxStdioTransport(
    #     package="mcp-time-server", 
    #     args=["-y"] 
    # )


    # # 2、Uvx调用
    # transport = UvxStdioTransport(
    #     tool_name="mcp-server-time",
    #     tool_args=["--local-timezone","UTC"],
    #     keep_alive=True
    # )

    # # 3、调用本地stdio文件，在本地运行对于python文件后，将对应python文件路径添加至此处
    # transport = "file.py"

    # # 4、调用本地sse文件，在本地运行对于python文件后，将对应地址和端口号添加至此处
    # transport=SSETransport("http://127.0.0.1:8888/sse")

    # # 5、调用本地StreamableHttp，在本地运行对于python文件后，将对应地址和端口号添加至此处
    # transport=StreamableHttpTransport("http://127.0.0.1:8888/mcp")
    # 连接 SSE 服务（URL 必须包含 /sse 路径）
    async with Client(transport) as client:  
        # result = await client.call_tool("list_directory", arguments={"path":r"C:\Users\24853\Desktop"})
        result = await client.call_tool("get_current_time", arguments={"timezone":r"UTC"})
        print("远程调用结果:", result) 
        await transport.close()## npx和uvx需要
        # 给子进程结束时间
        await asyncio.sleep(0.2)## npx和uvx需要
asyncio.run(test_remote_tool())