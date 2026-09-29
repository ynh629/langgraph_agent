"""
v1_agent.py
===========
V1：基于 LangGraph 的电商智能客服 Agent。
将三类业务能力封装为工具（语义检索 / 精确商品查询 / 价格计算），
接入 ReAct 循环，并启用会话记忆(checkpointer)。

运行：
    python v1_agent.py "星耀 X1 手机打 9 折后多少钱？"

依赖 .env 中的 deepseek_api_key / LLM_BASE_URL（见你的 graph.py 用法）。
"""
import os
from dotenv import load_dotenv

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from tools import AGENT_TOOLS

load_dotenv()

# ---- 1. LLM（对齐你 graph.py 的配置） ----
llm = ChatOpenAI(
    model_name=os.getenv("LLM_MODEL", "deepseek-flash"),
    temperature=0,
    max_tokens=1500,
    api_key=os.getenv("deepseek_api_key"),
    base_url=os.getenv("LLM_BASE_URL"),
)
llm_with_tools = llm.bind_tools(AGENT_TOOLS)

SYS_MSG = SystemMessage(
    content=(
        "你是电商平台智能客服助手，为顾客解答售前与售后问题。"
        "你有三个工具可用，请根据问题类型选择合适的工具：\n"
        "1. search_product_knowledge：语义检索知识库，用于退换货政策、优惠券规则、售后流程、物流配送等需要理解语义的问题；\n"
        "2. get_product_detail：精确查询某商品的价格、库存、规格、保修、退换条件等结构化信息；\n"
        "3. calculate：计算折扣后价格、满减后实付、差价等涉及数字运算的问题。\n"
        "回答必须严格基于工具返回的内容，不得编造商品信息或售后政策；"
        "若检索结果无法回答用户问题，请明确说明未检索到相关信息；"
        "涉及价格计算时先调用工具获取数据与计算，再给出结果。"
    )
)


# ---- 2. assistant 节点：LLM 生成回复或请求工具 ----
def assistant(state: MessagesState):
    response = llm_with_tools.invoke([SYS_MSG] + state["messages"])
    return {"messages": [response]}


# ---- 3. 构建 ReAct 图 ----
def build_react_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(AGENT_TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder


def main():
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "星耀 X1 手机打 9 折后多少钱？"

    graph = build_react_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "ecommerce-qa"}}

    print("=" * 60)
    print("用户问题：", question)
    print("=" * 60)

    # stream_mode="updates"：每完成一个节点就产出一次增量，用于逐步可视化 ReAct 轨迹
    for step, event in enumerate(
        graph.stream(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            stream_mode="updates",
        ),
        start=1,
    ):
        for node_name, update in event.items():
            if node_name == "assistant":
                msg = update["messages"][-1]
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        args = {k: (str(v)[:40] + "…" if len(str(v)) > 40 else v) for k, v in tc["args"].items()}
                        print(f"\n[步骤 {step}] assistant → 调用工具: {tc['name']}{args}")
                else:
                    print(f"\n[步骤 {step}] assistant → 最终回答:")
                    print(f"{msg.content}\n")
            elif node_name == "tools":
                for m in update["messages"]:
                    if isinstance(m, ToolMessage):
                        preview = m.content if isinstance(m.content, str) else str(m.content)
                        print(f"        ↳ 工具返回: {preview[:180]}{'…' if len(preview) > 180 else ''}")


if __name__ == "__main__":
    main()