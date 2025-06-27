# 从langchain.tools.retriever模块中导入create_retriever_tool函数，用于创建检索工具
from langchain.tools.retriever import create_retriever_tool
# 导入os模块，用于处理文件和目录路径
import os

# 从utils模块中导入get_embedding_model函数，用于获取嵌入模型
from utils import get_embedding_model

# 定义一个函数get_naive_rag_tool，接收一个参数vectorstore_name，表示向量存储的名称
def get_naive_rag_tool(vectorstore_name):
    # 从langchain_chroma模块中导入Chroma类，用于创建和管理向量数据库
    from langchain_chroma import Chroma

    # # 将数据添加到向量数据库中
    # vectorstore = Chroma(
    #     # 设置集合名称为传入的vectorstore_name
    #     collection_name=vectorstore_name,
    #     # 使用get_embedding_model函数获取嵌入模型，平台类型为"OpenAI"
    #     embedding_function=get_embedding_model(platform_type="gte_Qwen2-1.5B-instruct"),
    #     # 设置持久化目录，路径为当前文件的上上级目录下的"kb"文件夹中的vectorstore_name文件夹中的"vectorstore"文件夹
    #     persist_directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", vectorstore_name, "vectorstore"),
    # )

    vectorstore = Chroma(
        persist_directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", vectorstore_name, "vectorstore"),  # 允许我们将persist_directory目录保存到磁盘上
        # embedding_function=get_embedding_model(platform_type="gte_Qwen2-1.5B-instruct")
        embedding_function=get_embedding_model(platform_type="")
    )

    # 将向量数据库转换为检索器
    retriever = vectorstore.as_retriever(
        # 设置检索类型为"similarity_score_threshold"，表示使用相似度分数阈值进行检索
        search_type="similarity_score_threshold",
        # 设置检索参数，k为10表示返回前10个结果，score_threshold为0.15表示相似度分数的阈值为0.15
        search_kwargs={
            "k": 3,
            "score_threshold": 0.15,
        }
    )

    # 创建检索工具
    retriever_tool = create_retriever_tool(
        # 使用上面创建的检索器
        retriever,
        # 设置工具名称为"{vectorstore_name}_knowledge_base_tool"
        f"{vectorstore_name}_knowledge_base_tool",
        # 设置工具描述为"search and return information about {vectorstore_name}"
        f"search and return information about {vectorstore_name}",
    )
    # 设置检索工具的响应格式为"content"
    retriever_tool.response_format = "content"
    # 定义检索工具的功能函数，接收查询参数query
    retriever_tool.func = lambda query: {
        # 对检索结果进行处理，返回一个字典，键为"已知内容 {inum+1}"，值为文档内容
        f"已知内容 {inum+1}": doc.page_content.replace(doc.metadata["source"] + "\n\n", "")
        # 遍历检索器返回的文档，使用enumerate获取索引和文档
        for inum, doc in enumerate(retriever.invoke(query))
    }
    # 返回创建的检索工具
    return retriever_tool

# 如果当前模块是主程序入口
if __name__ == "__main__":
    # 调用get_naive_rag_tool函数，传入"personal_information"作为参数，获取检索工具
    retriever_tool = get_naive_rag_tool("personal_information")
    # 打印检索工具对查询"刘虔"的响应结果
    print(retriever_tool.invoke("刘虔"))