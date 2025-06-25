'''
1、创建fastmcp的实例
2、创建函数，添加文档
3、@mcp.tool
4、运行服务器
'''

from fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool()
def get_weather(city: str) -> str:
    '''
    获取对应城市天气
    :param city: 城市名称
    :return: 对应城市天气信息
    '''
    # 这里可以添加实际的天气查询逻辑
    return f"当前{city}的天气是晴天，温度25°C"

if __name__ == "__main__":
    mcp.run(transport='stdio')