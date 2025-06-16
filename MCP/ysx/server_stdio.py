from mcp.server.fastmcp import FastMCP
import os
mcp = FastMCP("FileSystem")

@mcp.tool()
def list_directory(path: str) -> list:
    """列出指定目录下的文件和子目录"""
    return os.listdir(path)

# @mcp.tool()
# def read_file(filepath: str) -> str:
#     """读取指定文件的内容"""
#     with open(filepath, 'r', encoding='utf-8') as f:
#         return f.read()

if __name__ == "__main__":
    mcp.run(transport="stdio")

# import os
# print(os.listdir("/Users/bytedance/Desktop/python/MCP/ysx"))