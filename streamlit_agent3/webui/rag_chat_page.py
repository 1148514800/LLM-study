# 导入streamlit库，用于创建Web应用
import streamlit as st
# 从utils模块中导入一些工具和函数
from utils import PLATFORMS, get_llm_models, get_chatllm, get_kb_names, get_img_base64
# 导入AI消息块和工具消息类，用于处理消息
from langchain_core.messages import AIMessageChunk, ToolMessage
# 导入内存保存器，用于保存会话状态
from langgraph.checkpoint.memory import MemorySaver
# 导入状态图和消息状态类，用于构建对话流程
from langgraph.graph import StateGraph, MessagesState
# 导入预构建的工具节点和工具条件，用于工具调用
from langgraph.prebuilt import ToolNode, tools_condition
# 从tools模块中导入获取简单RAG工具的函数
from tools import get_naive_rag_tool
# 导入json库，用于处理JSON数据
import json

# 定义RAG页面的介绍信息
RAG_PAGE_INTRODUCTION = "你好，我是逻辑推理增强后的大模型，请问有什么可以帮助你的吗？"

# 定义函数用于获取RAG图
def get_rag_graph(platform, model, temperature, selected_kbs, KBS):
    # 根据选择的知识库从KBS字典中获取工具
    tools = [KBS[k] for k in selected_kbs]
    # 创建工具节点
    tool_node = ToolNode(tools)

    # 定义调用模型的内部函数
    def call_model(state):
        # 获取聊天大模型
        llm = get_chatllm(platform, model, temperature=temperature)
        # 将工具绑定到模型
        llm_with_tools = llm.bind_tools(tools)
        # 返回调用结果
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # 创建状态图
    workflow = StateGraph(MessagesState)

    # 添加节点到状态图
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # 添加条件边和普通边
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    # 设置入口点
    workflow.set_entry_point("agent")

    # 创建内存保存器
    checkpointer = MemorySaver()
    # 编译应用
    app = workflow.compile(checkpointer=checkpointer)
    # graph_png = app.get_graph().draw_mermaid_png()
    # with open("./img/langgraph_rag", "wb") as f:
    #     f.write(graph_png)
    # 返回应用
    return app

# 定义函数用于处理图的响应
def graph_response(graph, input):
    # 调用图并处理事件
    for event in graph.invoke(
            {"messages": input},
            config={"configurable": {"thread_id": 42}},
            stream_mode="messages",
    ):
        # 如果事件是AI消息块
        if type(event[0]) == AIMessageChunk:
            # 如果有工具调用
            if len(event[0].tool_calls):
                # 添加工具调用信息到会话状态
                st.session_state["rag_tool_calls"].append(
                    {
                        "status": "正在查询...",
                        "knowledge_base": event[0].tool_calls[0]["name"].replace("_knowledge_base_tool", ""),
                        "query": ""
                    }
                )
            # 生成事件内容
            yield event[0].content
        # 如果事件是工具消息
        elif type(event[0]) == ToolMessage:
            # 创建状态占位符
            status_placeholder = st.empty()
            # 显示查询状态
            with (status_placeholder.status("正在查询...", expanded=True) as s):
                # 显示调用的知识库
                st.write("已调用 `", event[0].name.replace("_knowledge_base_tool", ""),
                         "` 知识库进行查询")  # 显示调用的工具
                # 初始化继续保存标志
                continue_save = False
                # 检查是否需要继续保存
                if len(st.session_state["rag_tool_calls"]):
                    if "content" not in st.session_state["rag_tool_calls"][-1].keys():
                        continue_save = True
                # 显示知识库检索结果
                st.write("知识库检索结果：")
                st.code(event[0].content, wrap_lines=True)
                # 更新状态
                s.update(label="已完成知识库检索！", expanded=False)
            # 如果需要继续保存
            if continue_save:
                st.session_state["rag_tool_calls"][-1]["status"] = "已完成知识库检索！"
                st.session_state["rag_tool_calls"][-1]["content"] = json.loads(event[0].content)
            else:
                # 添加新的工具调用信息
                st.session_state["rag_tool_calls"].append(
                    {
                        "status": "已完成知识库检索！",
                        "knowledge_base": event[0].name.replace("_knowledge_base_tool", ""),
                        "content": json.loads(event[0].content)
                    })

# 定义函数用于获取RAG聊天响应
def get_rag_chat_response(platform, model, temperature, input, selected_tools, KBS):
    # 获取RAG图
    app = get_rag_graph(platform, model, temperature, selected_tools, KBS)
    # 返回图的响应
    return graph_response(graph=app, input=input)

# 定义函数用于显示聊天历史
def display_chat_history():
    # 遍历聊天历史中的每条消息
    for message in st.session_state["rag_chat_history_with_tool_call"]:
        # 显示聊天消息
        with st.chat_message(message["role"],
                             avatar=get_img_base64("chatchat_avatar.png") if message["role"] == "assistant" else None):
            # 如果消息中有工具调用
            if "tool_calls" in message.keys():
                # 遍历工具调用
                for tool_call in message["tool_calls"]:
                    # 显示工具调用状态
                    with st.status(tool_call["status"], expanded=False):
                        st.write("已调用 `", tool_call["knowledge_base"], "` 知识库进行查询")
                        st.write("知识库检索结果：")
            # 显示消息内容
            st.write(message["content"])

# 定义函数用于清除聊天历史
def clear_chat_history():
    # 重置聊天历史
    st.session_state["rag_chat_history"] = [
        {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}
    ]
    st.session_state["rag_chat_history_with_tool_call"] = [
        {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}
    ]
    st.session_state["rag_tool_calls"] = []

# 定义RAG聊天页面函数
def rag_chat_page():
    # 获取知识库名称    现在只有一个bank知识库
    kbs = get_kb_names()
    # 创建知识库字典
    KBS = dict()
    # 遍历知识库名称并获取工具
    for k in kbs:
        KBS[f"{k}"] = get_naive_rag_tool(k)

    # 初始化会话状态中的聊天历史
    if "rag_chat_history" not in st.session_state:
        st.session_state["rag_chat_history"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}
        ]
    if "rag_chat_history_with_tool_call" not in st.session_state:
        st.session_state["rag_chat_history_with_tool_call"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}
        ]
    if "rag_tool_calls" not in st.session_state:
        st.session_state["rag_tool_calls"] = []

    # 在侧边栏中选择知识库
    with st.sidebar:
        selected_kbs = st.multiselect("请选择对话中可使用的知识库", kbs, default=kbs)

    # 显示聊天历史
    display_chat_history()

    # 在页面底部创建输入框和按钮
    with st._bottom:
        cols = st.columns([1.2, 10, 1])
        # 配置模型的弹出框
        with cols[0].popover(":gear:", use_container_width=True, help="配置模型"):
            platform = st.selectbox("请选择要使用的模型加载方式", PLATFORMS)
            # print("platform:", platform)
            model = st.selectbox("请选择要使用的模型", get_llm_models(platform))
            # print("model:", model)
            temperature = st.slider("请选择模型 Temperature", 0.1, 1., 0.1)
            history_len = st.slider("请选择历史消息长度", 1, 10, 5)
        # 输入框用于输入用户问题
        input = cols[1].chat_input("请输入您的问题")
        # 按钮用于清空对话
        cols[2].button(":wastebasket:", help="清空对话", on_click=clear_chat_history)
    # 如果有输入
    if input:
        # 显示用户消息
        with st.chat_message("user"):
            st.write(input)
        # 更新会话状态中的聊天历史
        st.session_state["rag_chat_history"] += [{"role": 'user', "content": input}]
        st.session_state["rag_chat_history_with_tool_call"] += [{"role": 'user', "content": input}]

        # 获取RAG聊天响应
        stream_response = get_rag_chat_response(
            platform,
            model,
            temperature,
            st.session_state["rag_chat_history"][-history_len:],
            selected_kbs,
            KBS
        )

        # 显示助手消息
        with st.chat_message("assistant", avatar=get_img_base64("chatchat_avatar.png")):
            response = st.write_stream(stream_response)
        # 更新会话状态中的聊天历史
        st.session_state["rag_chat_history"] += [{"role": 'assistant', "content": response}]
        st.session_state["rag_chat_history_with_tool_call"] += [
            {"role": 'assistant', "content": response, "tool_calls": st.session_state["rag_tool_calls"]}]
        # 清空工具调用记录
        st.session_state["rag_tool_calls"] = []
