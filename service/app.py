"""
service/app.py
==============
V4 最小可演示部署：FastAPI 暴露 V3 多智能体能力，含 SSE 流式、人工审批 resume、
输入/输出护栏与可观测性摘要。

接口：
  GET  /ok           健康检查（含工具数与 LLM 配置的就绪状态）
  GET  /tools        列出各专员可用的 MCP 工具（按售前/售后分组，含名称与说明）
  GET  /             轻量演示页（表单 + 流式输出）
  POST /chat         一次问答（非流式），返回最终答复 + trace 摘要
  POST /chat/stream  SSE 流式：逐节点推送路由/工具/回答，最后推送 answer 与 trace
  POST /resume       人工审批中断后恢复执行（approve / reject）

启动：
  python -m uvicorn service.app:app --host 0.0.0.0 --port 8000
"""
import json
import os
import sys
import uuid
from contextlib import AsyncExitStack, asynccontextmanager

# 离线优先：本地已缓存 BGE 模型，跳过 HuggingFace 联网检查（须在导入检索链之前）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_AGENT_DIR, os.path.join(_AGENT_DIR, "检索")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel

import order_store
from guardrails import detect_injection, guard_tools, mask_pii
from mcp_client import MCPToolHub
from observability import build_callbacks
from observability.tracing import TraceRecorder
from v3_agent import build_v3_graph

load_dotenv()

_BLOCKED_REPLY = "抱歉，您的消息触发了安全策略，已被拦截。如有正常咨询请换一种表达。"
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# ============================================================
# 请求模型
# ============================================================
class ChatRequest(BaseModel):
    message: str
    user_id: str = "U001"
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    decision: str  # approve / reject


class Runtime:
    """进程内常驻的图与 MCP 会话。"""

    def __init__(self, graph, hub, store) -> None:
        self.graph = graph
        self.hub = hub
        self.store = store


# ============================================================
# 序列化：把 LangChain 消息 / 节点更新转成 JSON 友好结构
# ============================================================
def _msg_to_dict(message) -> dict:
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    data = {"role": getattr(message, "type", type(message).__name__), "content": content}
    calls = getattr(message, "tool_calls", None)
    if calls:
        data["tool_calls"] = [{"name": c.get("name"), "args": c.get("args")} for c in calls]
    return data


def _update_to_event(node: str, update) -> dict:
    if not isinstance(update, dict):
        return {"node": node}
    event = {"node": node}
    if node == "supervisor":
        event["intent"] = update.get("intent")
    if update.get("messages"):
        event["messages"] = [_msg_to_dict(m) for m in update["messages"]]
    if update.get("approval_note"):
        event["approval_note"] = update["approval_note"]
    return event


def _final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            content = message.content
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _run(graph, graph_input, config):
    """驱动图并逐事件产出：{'type': 'node'|'interrupt', ...}。"""
    async for event in graph.astream(graph_input, config=config, stream_mode="updates"):
        if "__interrupt__" in event:
            for item in event["__interrupt__"]:
                yield {"type": "interrupt", "data": getattr(item, "value", item)}
            continue
        for node, update in event.items():
            yield {"type": "node", **_update_to_event(node, update)}


def _config(thread_id: str, callbacks: list) -> dict:
    return {"configurable": {"thread_id": thread_id}, "callbacks": callbacks}


def _payload(message: str, user_id: str) -> dict:
    return {"messages": [HumanMessage(content=message)], "user_id": user_id, "turns": 0}


# ============================================================
# 生命周期：常驻 MCP 会话 + 编译图
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    stack = AsyncExitStack()
    try:
        order_store.reset_orders()
    except OSError as exc:  # 只读文件系统下不阻断启动
        print(f"[warn] 订单数据重置失败（继续启动）：{exc}", file=sys.stderr)

    hub = await stack.enter_async_context(MCPToolHub())
    store = InMemoryStore()
    store.put(
        ("memory", "U001"),
        "profile",
        {"name": "张先生", "level": "黄金会员", "prefer": ["数码", "家电"]},
    )
    # 工具执行前做入参校验（护栏），再交给图
    graph = build_v3_graph(
        guard_tools(hub.presale_tools), guard_tools(hub.aftersale_tools)
    ).compile(checkpointer=MemorySaver(), store=store)

    app.state.runtime = Runtime(graph, hub, store)
    try:
        yield
    finally:
        await stack.aclose()


app = FastAPI(title="V4 电商客服多智能体服务", version="4.0", lifespan=lifespan)


# ============================================================
# 路由
# ============================================================
@app.get("/ok")
async def ok() -> dict:
    runtime: Runtime = getattr(app.state, "runtime", None)
    ready = runtime is not None
    return {
        "status": "ok" if ready else "starting",
        "tools": len(runtime.hub.all_tools) if ready else 0,
        "llm_model": os.getenv("LLM_MODEL", "deepseek-flash"),
    }


# 工具的中文展示名（英文名是机器标识，前端展示用中文；未登记的原样显示英文名）
_TOOL_LABELS = {
    "search_product_knowledge": "知识检索（商品/售后政策）",
    "get_product_detail": "商品详情查询",
    "calculate": "计算器（价格/折扣）",
    "get_order_status": "订单状态查询",
    "get_logistics": "物流轨迹查询",
    "apply_refund": "提交退款",
}


@app.get("/tools")
async def tools() -> dict:
    """列出各专员可用的 MCP 工具，按「售前 / 售后」分组（中文名 + 说明）。"""
    runtime: Runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return {"ready": False, "presale": [], "aftersale": []}

    def _describe(items) -> list:
        out = []
        for t in items:
            name = getattr(t, "name", "")
            out.append(
                {
                    "name": name,
                    "label": _TOOL_LABELS.get(name, name),
                    "description": (getattr(t, "description", "") or "").strip(),
                }
            )
        return out

    return {
        "ready": True,
        "presale": _describe(runtime.hub.presale_tools),
        "aftersale": _describe(runtime.hub.aftersale_tools),
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>V4 电商客服多智能体服务</h1><p>见 /docs 与 /ok</p>"


@app.post("/chat")
async def chat(req: ChatRequest) -> JSONResponse:
    runtime: Runtime = app.state.runtime
    thread_id = req.thread_id or uuid.uuid4().hex[:12]

    guard = detect_injection(req.message)
    if not guard.allowed:
        return JSONResponse(
            {
                "thread_id": thread_id,
                "blocked": True,
                "guard": {"category": guard.category, "reason": guard.reason},
                "answer": _BLOCKED_REPLY,
            }
        )

    recorder = TraceRecorder(thread_id=thread_id)
    callbacks = build_callbacks(thread_id=thread_id)
    callbacks[0] = recorder  # 保留本进程内的 recorder 以便读取摘要

    interrupt_payload = None
    async for event in _run(runtime.graph, _payload(req.message, req.user_id), _config(thread_id, callbacks)):
        if event["type"] == "interrupt":
            interrupt_payload = event["data"]

    snapshot = await runtime.graph.aget_state(_config(thread_id, callbacks))
    raw_answer = _final_answer(snapshot.values.get("messages", []))
    masked_answer, pii_hits = mask_pii(raw_answer)

    return JSONResponse(
        {
            "thread_id": thread_id,
            "blocked": False,
            "interrupted": interrupt_payload is not None,
            "interrupt": interrupt_payload,
            "answer": masked_answer,
            "pii_masked": pii_hits,
            "trace": recorder.summary(),
        }
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    runtime: Runtime = app.state.runtime
    thread_id = req.thread_id or uuid.uuid4().hex[:12]

    guard = detect_injection(req.message)
    if not guard.allowed:
        async def _blocked():
            yield _sse({"type": "blocked", "category": guard.category, "reason": guard.reason})
            yield _sse({"type": "answer", "thread_id": thread_id, "answer": _BLOCKED_REPLY})

        return StreamingResponse(_blocked(), media_type="text/event-stream")

    async def _generate():
        recorder = TraceRecorder(thread_id=thread_id)
        callbacks = build_callbacks(thread_id=thread_id)
        callbacks[0] = recorder
        config = _config(thread_id, callbacks)

        yield _sse({"type": "start", "thread_id": thread_id})
        async for event in _run(runtime.graph, _payload(req.message, req.user_id), config):
            yield _sse(event)

        snapshot = await runtime.graph.aget_state(config)
        masked_answer, pii_hits = mask_pii(_final_answer(snapshot.values.get("messages", [])))
        yield _sse(
            {
                "type": "answer",
                "thread_id": thread_id,
                "answer": masked_answer,
                "pii_masked": pii_hits,
                "trace": recorder.summary(),
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/resume")
async def resume(req: ResumeRequest) -> JSONResponse:
    runtime: Runtime = app.state.runtime
    recorder = TraceRecorder(thread_id=req.thread_id)
    callbacks = build_callbacks(thread_id=req.thread_id)
    callbacks[0] = recorder
    config = _config(req.thread_id, callbacks)

    interrupt_payload = None
    async for event in _run(runtime.graph, Command(resume=req.decision), config):
        if event["type"] == "interrupt":
            interrupt_payload = event["data"]

    snapshot = await runtime.graph.aget_state(config)
    masked_answer, pii_hits = mask_pii(_final_answer(snapshot.values.get("messages", [])))

    return JSONResponse(
        {
            "thread_id": req.thread_id,
            "decision": req.decision,
            "interrupted": interrupt_payload is not None,
            "interrupt": interrupt_payload,
            "answer": masked_answer,
            "pii_masked": pii_hits,
            "trace": recorder.summary(),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
