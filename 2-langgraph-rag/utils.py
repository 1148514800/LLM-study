# 导入os模块，用于操作系统相关的功能，如文件路径操作
import os
# 从typing模块导入Literal，用于类型注解，限制变量的取值范围
from typing import Literal
# 从langchain_openai模块导入ChatOpenAI类，用于与OpenAI的聊天模型进行交互
from langchain_openai import ChatOpenAI
# 从langchain_ollama模块导入ChatOllama类，用于与Ollama的聊天模型进行交互
from langchain_ollama import ChatOllama
# 从streamlit_flow模块导入streamlit_flow函数，用于在Streamlit中展示流程图
from streamlit_flow import streamlit_flow
# 从streamlit_flow.elements模块导入StreamlitFlowNode和StreamlitFlowEdge类，用于定义流程图中的节点和边
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
# 从streamlit_flow.state模块导入StreamlitFlowState类，用于管理流程图的状态
from streamlit_flow.state import StreamlitFlowState
# 从streamlit_flow.layouts模块导入TreeLayout类，用于定义流程图的布局
from streamlit_flow.layouts import TreeLayout
# 导入base64模块，用于进行base64编码和解码
import base64
# 从io模块导入BytesIO类，用于在内存中读写二进制数据
from io import BytesIO
# 导入streamlit模块，用于创建Web应用
import streamlit as st

# 定义一个常量PLATFORMS，包含支持的平台名称
PLATFORMS = ["OpenAI","Ollama", "Xinference", "Qwen2.5-7B-Instruct"] # ["fastchat"] "ZhipuAI",

# 定义一个函数get_llm_models，用于获取指定平台的LLM模型列表
def get_llm_models(platform_type: Literal[tuple(PLATFORMS)], base_url: str="", api_key: str="EMPTY"):
    # 如果平台类型是Ollama
    if platform_type == "Ollama":
        try:
            # 导入ollama模块
            import ollama
            # 如果没有提供base_url，使用默认的本地地址
            if not base_url:
                base_url = "http://127.0.0.1:11434"
            # 创建Ollama客户端
            client = ollama.Client(host=base_url)
            # 获取模型列表，排除包含"bert"的模型
            llm_models = [model["model"] for model in client.list()["models"] if "bert" not in model.details.families]
            # 返回模型列表
            return llm_models
        except Exception as e:
            # 如果发生异常，显示错误信息并返回空列表
            st.toast(f"尝试连接 {platform_type} 获取 LLM 模型时发生错误：\n{e}")
            return []
    # 如果平台类型是Xinference
    elif platform_type == "Xinference":
        try:
            # 从xinference_client模块导入RESTfulClient类并重命名为Client
            from xinference_client import RESTfulClient as Client
            # 如果没有提供base_url，使用默认的本地地址
            if not base_url:
                base_url = "http://127.0.0.1:9997"
            # 创建Xinference客户端
            client = Client(base_url=base_url)
            # 获取模型列表，筛选出类型为LLM的模型
            llm_models = client.list_models()
            return [k for k,v in llm_models.items() if v.get("model_type") == "LLM"]
        except Exception as e:
            # 如果发生异常，显示错误信息并返回空列表
            st.toast(f"尝试连接 {platform_type} 获取 LLM 模型时发生错误：\n{e}")
            return []
    # 如果平台类型是ZhipuAI
    elif platform_type == "ZhipuAI":
        # 这里是注释掉的代码，表示如何从ZhipuAI获取模型列表
        # from zhipuai import ZhipuAI
        #
        # client = ZhipuAI(
        #     api_key="",  # 填写您的 APIKey
        # )
        # client.list_models()
        # 返回一个预定义的模型列表
        return [
            'glm-4-alltools',
            'glm-4-plus',
            'glm-4-0520',
            'glm-4',
            'glm-4-air',
            'glm-4-airx',
            'glm-4-long',
            'glm-4-flashx',
            'glm-4-flash'
        ]
    # 使用qwen模型
    elif platform_type == "Qwen2.5-7B-Instruct":
        return [
            "Qwen2.5-7B-Instruct"
            ]
    # 如果平台类型是OpenAI
    elif platform_type == "OpenAI":
        # 这里是注释掉的代码，表示如何从OpenAI获取模型列表
        # from zhipuai import ZhipuAI
        #
        # client = ZhipuAI(
        #     api_key="",  # 填写您的 APIKey
        # )
        # client.list_models()
        # 返回一个预定义的模型列表
        return [
            'gpt-4o',
            'gpt-3.5-turbo'
        ]

# 定义一个函数get_embedding_models，用于获取指定平台的Embedding模型列表
def get_embedding_models(platform_type: Literal[tuple(PLATFORMS)], base_url: str="", api_key: str="EMPTY"):
    # 如果平台类型是Ollama
    if platform_type == "Ollama":
        try:
            # 导入ollama模块
            import ollama
            # 如果没有提供base_url，使用默认的本地地址
            if not base_url:
                base_url = "http://127.0.0.1:11434"
            # 创建Ollama客户端
            client = ollama.Client(host=base_url)
            # 获取模型列表，筛选出包含"bert"的模型
            embedding_models = [model["model"] for model in client.list()["models"] if "bert" in model.details.families]
            # 返回模型列表
            return embedding_models
        except Exception as e:
            # 如果发生异常，显示错误信息并返回空列表
            st.toast(f"尝试连接 {platform_type} 获取 Embedding 模型时发生错误：\n{e}")
            return []
    # 如果平台类型是Xinference
    elif platform_type == "Xinference":
        try:
            # 从xinference_client模块导入RESTfulClient类并重命名为Client
            from xinference_client import RESTfulClient as Client
            # 如果没有提供base_url，使用默认的本地地址
            if not base_url:
                base_url = "http://127.0.0.1:9997"
            # 创建Xinference客户端
            client = Client(base_url=base_url)
            # 获取模型列表，筛选出类型为embedding的模型
            embedding_models = client.list_models()
            return [k for k,v in embedding_models.items() if v.get("model_type") == "embedding"]
        except Exception as e:
            # 如果发生异常，显示错误信息并返回空列表
            st.toast(f"尝试连接 {platform_type} 获取 Embedding 模型时发生错误：\n{e}")
            return []
    # 如果平台类型是qwen
    elif platform_type == "gte_Qwen2-1.5B-instruct":
        try:
            embedding_models = ["gte_Qwen2-1.5B-instruct"]
            return embedding_models
        except Exception as e:
            # 如果发生异常，显示错误信息并返回空列表
            st.toast(f"尝试连接 {platform_type} 获取 Embedding 模型时发生错误：\n{e}")
            return []

# 定义一个函数get_chatllm，用于获取指定平台的聊天模型
def get_chatllm(
        platform_type: Literal[tuple(PLATFORMS)],
        model: str,
        base_url: str = "",
        api_key: str = "",
        temperature: float = 0.1
):
    # 如果平台类型是Ollama
    if platform_type == "Ollama":
        # 如果没有提供base_url，使用默认的本地地址
        if not base_url:
            base_url = "http://127.0.0.1:11434"
        # 返回一个ChatOllama对象，用于与Ollama的聊天模型交互
        return ChatOllama(
            temperature=temperature,
            # streaming=True,  # 这里是注释掉的代码，表示是否启用流式传输
            model=model,
            base_url=base_url
        )
    # 如果平台类型是Xinference
    elif platform_type == "Xinference":
        # 如果没有提供base_url，使用默认的本地地址
        if not base_url:
            base_url = "http://127.0.0.1:9997/v1"
        # 如果没有提供api_key，使用默认的"EMPTY"
        if not api_key:
            api_key = "EMPTY"
    # 如果平台类型是ZhipuAI
    elif platform_type == "ZhipuAI":
        # 如果没有提供base_url，使用默认的地址
        if not base_url:
            base_url = "https://open.bigmodel.cn/api/paas/v4"
        # 如果没有提供api_key，使用默认的"EMPTY"
        if not api_key:
            api_key = "EMPTY"
    # 如果平台类型是OpenAI
    elif platform_type == "OpenAI":
        # 如果没有提供base_url，使用默认的地址
        if not base_url:
            base_url = "https://api.openai.com/v1"
        # 如果没有提供api_key，从环境变量中获取
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')

    # 返回一个ChatOpenAI对象，用于与OpenAI的聊天模型交互
    return ChatOpenAI(
        temperature=temperature,
        model_name=model,
        streaming=True,
        base_url=base_url,
        api_key=api_key,
    )

# 定义一个函数show_graph，用于在Streamlit中展示流程图
def show_graph(graph):
    # 创建一个StreamlitFlowState对象，包含节点和边的信息
    flow_state = StreamlitFlowState(
                       # 创建节点列表，每个节点包含id、位置、数据和类型
                       nodes=[StreamlitFlowNode(
                           id=node.id,
                           pos=(0,0),
                           data={"content": node.id},
                           node_type="input" if node.id == "__start__"
                                             else "output" if node.id == "__end__"
                                             else "default",
                       ) for node in graph.nodes.values()],
                       # 创建边列表，每条边包含id、源节点、目标节点和动画效果
                       edges=[StreamlitFlowEdge(
                           id=str(enum),
                           source=edge.source,
                           target=edge.target,
                           animated=True,
                       ) for enum, edge in enumerate(graph.edges)],
                   )
    # 使用streamlit_flow函数展示流程图，指定布局和视图设置
    streamlit_flow('example_flow',
                   flow_state,
                   layout=TreeLayout(direction='down'), fit_view=True
    )

# 定义一个函数get_kb_names，用于获取知识库的名称列表
def get_kb_names():
    # 获取当前文件所在目录下的kb目录路径
    kb_root = os.path.join(os.path.dirname(__file__), "kb")
    # 如果kb目录不存在，则创建该目录
    if not os.path.exists(kb_root):
        os.mkdir(kb_root)
    # 获取kb目录下的所有子目录名称，并返回这些名称
    kb_names = [f for f in os.listdir(kb_root) if os.path.isdir(os.path.join(kb_root, f))]
    return kb_names

# 定义一个函数get_embedding_model，用于获取指定平台的Embedding模型
def get_embedding_model(
        platform_type: Literal[tuple(PLATFORMS)] = "Ollama",
        model: str = "text-embedding-ada-002",
        base_url: str = os.getenv('OPENAI_BASE_URL'),
        api_key: str = os.getenv('OPENAI_API_KEY'),
):
    # 如果平台类型是Ollama
    if platform_type == "Ollama":
        # 这里是注释掉的代码，表示如何使用ChatOllama
        # from langchain_ollama import ChatOllama
        # return ChatOllama
        # 如果没有提供base_url，使用默认的本地地址
        if not base_url:
            base_url = "http://127.0.0.1:11434/"
        # 从langchain_ollama模块导入OllamaEmbeddings类
        from langchain_ollama import OllamaEmbeddings
        # 返回一个OllamaEmbeddings对象，用于与Ollama的Embedding模型交互
        return OllamaEmbeddings(base_url=base_url, model=model)
    # 如果平台类型是Xinference
    elif platform_type == "Xinference":
        # 从langchain_community.embeddings.xinference模块导入XinferenceEmbeddings类
        from langchain_community.embeddings.xinference import XinferenceEmbeddings
        # 如果没有提供base_url，使用默认的本地地址
        if not base_url:
            base_url = "http://127.0.0.1:9997/v1"
        # 返回一个XinferenceEmbeddings对象，用于与Xinference的Embedding模型交互
        return XinferenceEmbeddings(server_url=base_url, model_uid=model)
    # 使用gte_Qwen2-1.5B-instruct
    elif platform_type == "gte_Qwen2-1.5B-instruct":
        # 3、加载向量模型
        from langchain.embeddings import HuggingFaceEmbeddings

        # 指定要加载的预训练模型的名称，参考排行榜：https://huggingface.co/spaces/mteb/leaderboard
        model_name = "/home/hpclp/disk/q/models/gte_Qwen2-1.5B-instruct"

        # 创建 Hugging Face 的嵌入模型实例，这个模型将用于将文本转换为向量表示（embedding）
        embedding_models = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cuda:0"})
        return embedding_models

    # 否则，使用OpenAI的Embedding模型
    else:
        # 从langchain_openai模块导入OpenAIEmbeddings类
        from langchain_openai import OpenAIEmbeddings
        # 返回一个OpenAIEmbeddings对象，用于与OpenAI的Embedding模型交互
        return OpenAIEmbeddings(base_url=base_url, api_key=api_key, model=model)

# 定义一个函数get_img_base64，用于获取图片的base64编码
def get_img_base64(file_name: str) -> str:
    """
    get_img_base64 used in streamlit.
    absolute local path not working on windows.
    """
    # 获取当前文件所在目录下的img目录中的图片路径
    image_path = os.path.join(os.path.dirname(__file__), "img", file_name)
    # 读取图片
    with open(image_path, "rb") as f:
        # 将图片数据读入内存
        buffer = BytesIO(f.read())
        # 对图片数据进行base64编码，并解码为字符串
        base_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    # 返回base64编码的图片数据，格式为data:image/png;base64,编码数据
    return f"data:image/png;base64,{base_str}"

# 如果当前模块是主程序入口
if __name__ == "__main__":
    # 打印Ollama平台的默认Embedding模型
    print(get_embedding_model(platform_type="Ollama"))
