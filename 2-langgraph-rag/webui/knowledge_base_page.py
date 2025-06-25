import os

# 导入streamlit库，用于创建Web应用
import streamlit as st
# 从utils模块中导入PLATFORMS, get_embedding_models, get_kb_names函数
from utils import PLATFORMS, get_embedding_models, get_kb_names
# 从langchain_chroma模块中导入Chroma类
from langchain_chroma import Chroma
# 从langchain_text_splitters模块中导入RecursiveCharacterTextSplitter, MarkdownTextSplitter类
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

# 从utils模块中导入get_embedding_model函数
from utils import get_embedding_model

# 定义知识库管理页面的函数
def knowledge_base_page():
    # 检查session_state中是否有"selected_kb"，如果没有则初始化为空字符串
    if "selected_kb" not in st.session_state:
        st.session_state["selected_kb"] = ""
    # 设置页面标题为“知识库管理”
    st.title("推理增强知识库")
    # 获取所有知识库的名称
    kb_names = get_kb_names()
    # 创建一个下拉选择框，供用户选择知识库
    selected_kb = st.selectbox("请选择知识库",
                               ["新建知识库"] + kb_names,
                               index=kb_names.index(st.session_state["selected_kb"]) + 1
                               if st.session_state["selected_kb"] in kb_names
                               else 0
                               )
    # 如果用户选择了“新建知识库”
    if selected_kb == "新建知识库":
        # 创建一个占位符，用于显示状态信息
        status_placeholder = st.empty()
        # 使用占位符创建一个状态信息框，显示“知识库配置”
        with status_placeholder.status("知识库配置", expanded=True) as s:
            # 创建两列布局
            cols = st.columns(2)
            # 在第一列中创建一个文本输入框，供用户输入知识库名称
            kb_name = cols[0].text_input("请输入知识库名称", placeholder="请使用英文，如：companies_information")
            # 在第二列中创建一个下拉选择框，供用户选择向量库类型
            vs_type = cols[1].selectbox("请选择向量库类型", ["Chroma"])
            # 创建一个文本区域，供用户输入知识库描述
            st.text_area("请输入知识库描述", placeholder="如：介绍企业基本信息")
            # 再次创建两列布局
            cols = st.columns(2)
            # 在第一列中创建一个下拉选择框，供用户选择Embedding模型加载方式
            platform = cols[0].selectbox("请选择要使用的 Embedding 模型加载方式", PLATFORMS)
            # 获取用户选择的平台对应的Embedding模型列表
            embedding_models = get_embedding_models(platform)   # 这里返回的只是一个列表
            # 在第二列中创建一个下拉选择框，供用户选择具体的Embedding模型
            embedding_model = cols[1].selectbox("请选择要使用的 Embedding 模型", embedding_models)
            # 创建一个按钮，供用户提交创建知识库的请求
            submit = st.button("创建知识库")
            # 如果用户点击了提交按钮且知识库名称不为空
            if submit and kb_name.strip():
                # 构建知识库的根目录路径
                kb_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb")
                # 构建具体知识库的路径
                kb_path = os.path.join(kb_root, kb_name)
                # 构建文件存储路径
                file_storage_path = os.path.join(kb_path, "files")
                # 构建向量存储路径
                vs_path = os.path.join(kb_path, "vectorstore")
                # 如果知识库路径不存在，则创建该目录
                if not os.path.exists(kb_path):
                    os.mkdir(kb_path)
                # 如果文件存储路径不存在，则创建该目录
                if not os.path.exists(file_storage_path):
                    os.mkdir(file_storage_path)
                # 如果向量存储路径不存在，则创建该目录
                if not os.path.exists(vs_path):
                    os.mkdir(vs_path)
                else:
                    # 如果知识库已存在，显示错误信息
                    st.error("知识库已存在")
                    # 更新状态信息框为错误状态
                    s.update(label=f'知识库配置', expanded=True, state="error")
                    # 停止执行
                    st.stop()
                # 显示创建成功的信息
                st.success("创建知识库成功")
                # 更新状态信息框为成功状态
                s.update(label=f'已创建知识库"{kb_name}"', expanded=False)
                # 更新session_state中的selected_kb为新创建的知识库名称
                st.session_state["selected_kb"] = kb_name
                # 重新运行应用
                st.rerun()
            # 如果用户点击了提交按钮但知识库名称为空
            elif submit and not kb_name.strip():
                # 显示错误信息
                st.error("知识库名称不能为空")
                # 更新状态信息框为错误状态
                s.update(label=f'知识库配置', expanded=True, state="error")
                # 停止执行
                st.stop()
    else:
        # 构建知识库的根目录路径
        kb_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb")
        # 构建具体知识库的路径
        kb_path = os.path.join(kb_root, selected_kb)
        # 构建文件存储路径
        file_storage_path = os.path.join(kb_path, "files")
        # 构建向量存储路径
        vs_path = os.path.join(kb_path, "vectorstore")
        # 创建一个占位符，用于文件上传
        uploader_placeholder = st.empty()

        # 支持的文件格式列表
        supported_file_formats = ["md"] #, "txt"
        # 使用占位符创建一个状态信息框，显示“上传文件至知识库”
        with uploader_placeholder.status("上传文件至知识库", expanded=True) as s:
            # 创建一个文件上传控件，支持多文件上传
            files = st.file_uploader("请上传文件", type=supported_file_formats, accept_multiple_files=True)
            # 创建一个按钮，供用户提交上传文件的请求
            upload = st.button("上传")
        # 如果用户点击了上传按钮
        if upload:
            # 遍历用户上传的文件
            for file in files:
                # 获取文件的二进制内容
                b = file.getvalue()
                # 将文件内容写入到指定路径
                with open(os.path.join(file_storage_path, file.name), "wb") as f:
                    f.write(b)

            # 从langchain_community.document_loaders模块中导入DirectoryLoader, TextLoader类
            from langchain_community.document_loaders import DirectoryLoader, TextLoader
            # 设置文本加载器的参数，自动检测编码
            text_loader_kwargs = {"autodetect_encoding": True}
            # 创建一个目录加载器，用于加载指定路径下的文件
            loader = DirectoryLoader(
                file_storage_path,
                glob=[f"**/{file.name}" for file in files],
                show_progress=True,
                use_multithreading=True,
                loader_cls=TextLoader,
                loader_kwargs=text_loader_kwargs,
            )
            # 加载文件并返回文档列表
            docs_list = loader.load()

            # 创建一个Markdown文本分割器，设置分块大小和重叠
            text_splitter = MarkdownTextSplitter(
                chunk_size=500, chunk_overlap=100
            )
            # 将文档列表分割成多个小块
            doc_splits = text_splitter.split_documents(docs_list)
            # 遍历分割后的文档块
            for doc in doc_splits:
                # 将文档的源信息添加到内容前面
                doc.page_content = doc.metadata["source"] + "\n\n" + doc.page_content

            # # 导入chromadb.api模块
            # import chromadb.api

            # # 清除Chroma数据库的系统缓存
            # chromadb.api.client.SharedSystemClient.clear_system_cache()

            # # 创建一个Chroma向量存储对象
            # vectorstore = Chroma(
            #     collection_name=selected_kb,
            #     embedding_function=get_embedding_model(platform_type="gte_Qwen2-1.5B-instruct"),
            #     persist_directory=vs_path,
            # )

            # # 将分割后的文档添加到向量存储中
            # vectorstore.add_documents(doc_splits)

            from langchain.vectorstores.chroma import Chroma
            vectorstore  = Chroma.from_documents(
                documents=doc_splits, # 
                embedding=get_embedding_model(platform_type="gte_Qwen2-1.5B-instruct"),
                persist_directory=vs_path  # 允许我们将persist_directory目录保存到磁盘上
            )
            vectorstore.persist()
            # 显示上传成功的信息
            st.success("上传文件成功")

