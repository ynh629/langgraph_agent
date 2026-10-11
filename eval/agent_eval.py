"""
eval/agent_eval.py
==================
Agent 编排评估：用「脚本化 LLM」驱动 V3/V4 图，对数据集中每个场景断言
【路由序列 / 工具调用 / 是否中断 / 最终话术要点 / 订单落盘状态】。

为什么用脚本化 LLM（面试可讲）：
  真实 LLM 有决策随机性，会把「图编排是否正确」和「模型发挥是否稳定」两件事混在一起。
  注入固定剧本后，本层评估变成确定性门禁，可稳定复现，适合做 CI 硬门禁。
"""
import asyncio
import json
import os
import sys

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_AGENT_DIR, os.path.join(_AGENT_DIR, "检索")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

import order_store
from mcp_client import MCPToolHub
from v3_agent import build_v3_graph

from .metrics import accuracy

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.jsonl")


class ScriptedLLM:
    """按固定剧本返回 AIMessage 的假 LLM，使图编排可确定性验证。"""

    def __init__(self, steps):
        self._steps = list(steps)
        self._index = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        if self._index >= len(self._steps):
            raise AssertionError("ScriptedLLM 剧本耗尽：图执行次数与预期不符")
        step = self._steps[self._index]
        self._index += 1
        kind = step.get("kind")
        if kind == "route":
            return AIMessage(content=step["target"])
        if kind == "tools":
            return AIMessage(content="", tool_calls=[dict(call) for call in step["calls"]])
        if kind == "reply":
            return AIMessage(content=step["text"])
        raise ValueError(f"未知剧本步骤：{kind}")


def load_dataset(path: str = DATASET) -> list[dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _ai_text(messages) -> str:
    parts = []
    for m in messages:
        if isinstance(m, AIMessage):
            content = getattr(m, "content", "")
            parts.append(content if isinstance(content, str) else str(content))
    return "\n".join(parts)


def _called_tools(messages) -> list:
    names = []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            names.append(call.get("name"))
    return names


async def _drive(graph, stream_input, config) -> tuple:
    """消费一次 astream，返回 (是否中断, 中断载荷, 本次 supervisor 路由序列)。"""
    interrupted = False
    interrupt_value = None
    routes = []
    async for event in graph.astream(stream_input, config=config, stream_mode="updates"):
        if "__interrupt__" in event:
            interrupted = True
            for item in event["__interrupt__"]:
                interrupt_value = getattr(item, "value", item)
            continue
        for node, update in event.items():
            if node == "supervisor" and isinstance(update, dict) and update.get("intent"):
                routes.append(update["intent"])
    return interrupted, interrupt_value, routes


async def run_case(graph, case: dict) -> tuple:
    config = {"configurable": {"thread_id": case["thread_id"]}}
    payload = {
        "messages": [HumanMessage(content=case["question"])],
        "user_id": case["user_id"],
        "turns": 0,
    }
    interrupted, interrupt_value, routes = await _drive(graph, payload, config)
    if interrupted and case.get("resume") is not None:
        _, _, more_routes = await _drive(graph, Command(resume=case["resume"]), config)
        routes += more_routes
    snapshot = await graph.aget_state(config)
    return interrupted, interrupt_value, routes, snapshot.values


def check_case(case: dict, interrupted: bool, routes: list, values: dict) -> dict:
    expect = case["expect"]
    failures = []
    aspects = {}

    aspects["route"] = routes == expect["routes"]
    if not aspects["route"]:
        failures.append(f"路由序列 {routes} != 期望 {expect['routes']}")

    tools = sorted(_called_tools(values["messages"]))
    aspects["tools"] = tools == sorted(expect["called_tools"])
    if not aspects["tools"]:
        failures.append(f"工具调用 {tools} != 期望 {sorted(expect['called_tools'])}")

    aspects["interrupt"] = bool(interrupted) == bool(expect["interrupted"])
    if not aspects["interrupt"]:
        failures.append(f"中断标志 {interrupted} != 期望 {expect['interrupted']}")

    text = _ai_text(values["messages"])
    aspects["final"] = all(token in text for token in expect["final_contains"])
    if not aspects["final"]:
        failures.append(f"最终话术缺少要点 {expect['final_contains']}｜实际：{text[-200:]}")

    aspects["state"] = True
    if expect.get("refund_order"):
        order = order_store.get_order(expect["refund_order"]) or {}
        got = order.get("refund_status")
        aspects["state"] = got == expect["refund_status"]
        if not aspects["state"]:
            failures.append(f"订单 {expect['refund_order']} 状态 {got} != 期望 {expect['refund_status']}")

    return {
        "id": case["id"],
        "title": case["title"],
        "category": case["category"],
        "ok": all(aspects.values()),
        "aspects": aspects,
        "failures": failures,
    }


async def run(dataset_path: str = DATASET) -> dict:
    cases = load_dataset(dataset_path)
    order_store.reset_orders()
    store = InMemoryStore()
    store.put(("memory", "U001"), "profile", {"name": "张先生", "level": "黄金会员", "prefer": ["数码", "家电"]})

    results = []
    async with MCPToolHub() as hub:
        for case in cases:
            scripted = ScriptedLLM(case["script"])
            graph = build_v3_graph(hub.presale_tools, hub.aftersale_tools, llm_client=scripted).compile(
                checkpointer=MemorySaver(), store=store
            )
            interrupted, _, routes, values = await run_case(graph, case)
            results.append(check_case(case, interrupted, routes, values))

    total = len(results) or 1
    summary = {
        "total": len(results),
        "task_success": round(sum(r["ok"] for r in results) / total, 4),
        "route_accuracy": accuracy([r["aspects"]["route"] for r in results]),
        "tool_accuracy": accuracy([r["aspects"]["tools"] for r in results]),
        "interrupt_accuracy": accuracy([r["aspects"]["interrupt"] for r in results]),
        "state_accuracy": accuracy([r["aspects"]["state"] for r in results]),
    }
    return {"layer": "agent", "summary": summary, "cases": results}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), ensure_ascii=False, indent=2))
