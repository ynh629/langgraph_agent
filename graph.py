"""
LangGraph 核心功能完整示例
============================
涵盖：
  1. ReAct 核心框架（assistant + tools）
  2. 自定义 State 与 Reducer
  3. RemoveMessage 消息裁剪
  4. 静态并行（fan-out / fan-in）
  5. 动态并行（Send API，map-reduce）
  6. 两套记忆系统（checkpointer + store）
  7. 静态断点（interrupt_before）
  8. 动态断点（interrupt + Command）

"""

from typing import Annotated, TypedDict
import operator

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, RemoveMessage
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Send, interrupt, Command
from dotenv import load_dotenv
import os


# ============================================================
# 0. 初始化 LLM
# ============================================================
load_dotenv()
llm = ChatOpenAI(model_name="deepseek-chat",temperature=0,max_tokens=1000,api_key=os.getenv("deepseek_api_key"),base_url=os.getenv("LLM_BASE_URL"))


# ============================================================
# 1. 定义工具
# ============================================================
@tool
def add(a: int, b: int) -> int:
    """两数相加"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """两数相乘"""
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """两数相除"""
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


tools = [add, multiply, divide]
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="你是一个能进行基础算术运算的助手。")


# ============================================================
# 2. ReAct 核心框架
# ============================================================
def assistant(state: MessagesState):
    """assistant 节点：调用 LLM 生成回复或请求工具"""
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}


def build_react_graph():
    """构建标准 ReAct 图：assistant ↔ tools"""
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder


# ============================================================
# 3. 自定义 State 与 Reducer
# ============================================================
class CustomState(TypedDict):
    messages: Annotated[list, add_messages]   # 消息：追加 + 去重
    docs: Annotated[list, operator.add]       # 列表：拼接
    counter: int                              # 无注解：覆盖


def delete_old_messages(state: CustomState):
    """保留最近 2 条消息，其余用 RemoveMessage 删除"""
    return {
        "messages": [
            RemoveMessage(id=m.id)
            for m in state["messages"][:-2]
        ]
    }


# ============================================================
# 4. 静态并行（fan-out / fan-in）
# ============================================================
class ParallelState(TypedDict):
    results: Annotated[list[str], operator.add]


def node_a(state): return {"results": ["a"]}
def node_b(state): return {"results": ["b"]}
def node_c(state): return {"results": ["c"]}
def node_d(state): return {"results": ["d"]}


def build_parallel_graph():
    """a → (b, c) → d 的静态并行"""
    builder = StateGraph(ParallelState)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_node("c", node_c)
    builder.add_node("d", node_d)

    builder.add_edge(START, "a")
    builder.add_edge("a", "b")      # fan-out
    builder.add_edge("a", "c")
    builder.add_edge("b", "d")      # fan-in
    builder.add_edge("c", "d")
    builder.add_edge("d", END)

    return builder


# ============================================================
# 5. 动态并行（Send API，map-reduce）
# ============================================================
class OverallState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str


def generate_topics(state: OverallState):
    """生成多个子主题"""
    return {"subjects": ["程序员", "猫", "咖啡"]}


def continue_to_jokes(state: OverallState):
    """条件边返回 Send 列表 → 每个 subject 一个并行分支"""
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


def generate_joke(state: dict):
    """每个分支独立执行"""
    return {"jokes": [f"关于{state['subject']}的笑话"]}


def best_joke(state: OverallState):
    """reduce 阶段：选出最佳"""
    return {"best_selected_joke": state["jokes"][0]}


def build_map_reduce_graph():
    """动态并行 map-reduce 图"""
    builder = StateGraph(OverallState)
    builder.add_node("generate_topics", generate_topics)
    builder.add_node("generate_joke", generate_joke)
    builder.add_node("best_joke", best_joke)

    builder.add_edge(START, "generate_topics")
    builder.add_conditional_edges(
        "generate_topics",
        continue_to_jokes,
        ["generate_joke"],
    )
    builder.add_edge("generate_joke", "best_joke")
    builder.add_edge("best_joke", END)

    return builder


# ============================================================
# 6. 动态断点节点
# ============================================================
def assistant_with_approval(state: MessagesState):
    """在工具调用前用 interrupt() 暂停，等待人工审批"""
    response = llm_with_tools.invoke([sys_msg] + state["messages"])

    if response.tool_calls:
        decision = interrupt({
            "question": "是否允许调用工具？",
            "tool_calls": response.tool_calls,
        })
        if decision != "yes":
            return {"messages": [AIMessage(content="已取消工具调用")]}

    return {"messages": [response]}


def build_dynamic_interrupt_graph():
    """动态断点图"""
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant_with_approval)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder


# ============================================================
# 7. 演示函数
# ============================================================
def demo_react():
    print("\n" + "=" * 60)
    print("【1】ReAct 核心框架")
    print("=" * 60)

    graph = build_react_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "react-1"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="3 加 5 等于多少？再乘以 2")]},
        config=config,
        )
    print("最终回复：", result["messages"][-1].content)


def demo_parallel():
    print("\n" + "=" * 60)
    print("【2】静态并行")
    print("=" * 60)

    graph = build_parallel_graph().compile()
    result = graph.invoke({"results": []})
    print("结果：", result["results"])


def demo_map_reduce():
    print("\n" + "=" * 60)
    print("【3】动态并行 map-reduce")
    print("=" * 60)

    graph = build_map_reduce_graph().compile()
    result = graph.invoke({"topic": "幽默"})
    print("所有笑话：", result["jokes"])
    print("最佳笑话：", result["best_selected_joke"])


def demo_memory():
    print("\n" + "=" * 60)
    print("【4】两套记忆系统")
    print("=" * 60)

    across_thread_memory = InMemoryStore()
    within_thread_memory = MemorySaver()

    graph = build_react_graph().compile(
        checkpointer=within_thread_memory,
        store=across_thread_memory,
    )

    config = {"configurable": {"thread_id": "mem-1", "user_id": "1"}}

    # 长期记忆：写入用户画像
    user_id = "1"
    across_thread_memory.put(
        ("memory", user_id),
        "user_profile",
        {"name": "小明", "interests": ["AI", "LangGraph"]},
    )

    item = across_thread_memory.get(("memory", user_id), "user_profile")
    print("长期记忆读取：", item.value)

    # 短期记忆：多轮对话
    graph.invoke(
        {"messages": [HumanMessage(content="我叫小明，喜欢 AI")]},
        config=config,
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="我叫什么？")]},
        config=config,
    )
    print("短期记忆回复：", result["messages"][-1].content)


def demo_static_interrupt():
    print("\n" + "=" * 60)
    print("【5】静态断点（自动模拟审批 yes）")
    print("=" * 60)

    graph = build_react_graph().compile(
        checkpointer=MemorySaver(),
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "static-1"}}

    graph.invoke(
        {"messages": [HumanMessage(content="3 加 5 等于多少？")]},
        config=config,
    )
    print("已暂停在 tools 节点前，等待审批...")

    # 模拟人类批准
    print("模拟审批：yes")
    result = None
    for event in graph.stream(None, config=config, stream_mode="values"):
        result = event
    print("恢复后最终回复：", result["messages"][-1].content)


def demo_dynamic_interrupt():
    print("\n" + "=" * 60)
    print("【6】动态断点（自动模拟审批 yes）")
    print("=" * 60)

    graph = build_dynamic_interrupt_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "dynamic-1"}}

    graph.invoke(
        {"messages": [HumanMessage(content="3 加 5 等于多少？")]},
        config=config,
    )
    print("已暂停在 interrupt() 处，等待审批...")

    # 模拟人类批准
    print("模拟审批：yes")
    result = None
    for event in graph.stream(
        Command(resume="yes"), config=config, stream_mode="values"
    ):
        result = event
    print("恢复后最终回复：", result["messages"][-1].content)


# ============================================================
# 8. 主入口
# ============================================================
if __name__ == "__main__":
    demo_react()
    demo_parallel()
    demo_map_reduce()
    demo_memory()
    demo_static_interrupt()
    demo_dynamic_interrupt()

    print("\n" + "=" * 60)
    print("✅ 所有核心功能演示完成")
    print("=" * 60)