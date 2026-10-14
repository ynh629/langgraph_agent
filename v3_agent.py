"""
v3_agent.py
===========
V3：电商客服多智能体系统（MCP 版）—— 在 V2 的图结构上，把「工具来源」从
进程内 LangChain 工具换成「跨进程的 MCP Server」，并把整条链路异步化。

与 V2 的差异（面试可讲）：
  1. 工具来源：V2 = tools.py 进程内 @tool；V3 = 两个 MCP Server（stdio）
     经 MCPToolHub 动态发现并按白名单分组，工具是「远程能力」而非本地函数。
  2. 执行模型：V2 = 同步 invoke/stream；V3 = 异步 ainvoke/astream，
     因为 MCP 工具本质是 async，且异步更契合真实服务调用。
  3. 图结构保持一致：supervisor 路由 + 三专员 + approval 审批 + 两个工具节点，
     提示词与 V2 相同，保证 V3 与 V2 输出语义一致。

服务边界（对应两个 MCP Server）：
    retrieval（检索服务）→ search_product_knowledge / get_product_detail / calculate
    order（订单服务）    → get_order_status / get_logistics / apply_refund
权限隔离：售前专员只绑定检索服务中的商品类工具，结构上拿不到订单与退款工具。

运行：
    python v3_agent.py          # 依次跑 4 个演示场景
    python v3_agent.py 2        # 只跑第 2 个场景
"""
import asyncio
import json
import os
import sys

# 离线优先：本地已缓存 BGE 模型，跳过 HuggingFace 联网检查（必须在导入检索链之前）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt

import order_store
from mcp_client import MCPToolHub
from policy import (
    REFUND_APPROVAL_THRESHOLD,
    extract_refund_request,
    needs_human_approval,
)

load_dotenv()

# 路由次数上限，防止 supervisor 与专员之间反复横跳导致死循环
MAX_SUPERVISOR_TURNS = 8


# ============================================================
# 1. LLM（supervisor / ticket 使用；专员用的 bind_tools 版本在 build 时生成）
#    懒加载：确定性回归测试（注入假 LLM）时无需构造真实客户端，也不依赖 .env
# ============================================================
_llm_singleton = None


def get_llm() -> ChatOpenAI:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = ChatOpenAI(
            model_name=os.getenv("LLM_MODEL", "deepseek-flash"),
            temperature=0,
            max_tokens=1500,
            api_key=os.getenv("deepseek_api_key"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
    return _llm_singleton


# ============================================================
# 2. 共享 State（与 V2 完全一致）
# ============================================================
class AgentState(MessagesState):
    user_id: str
    intent: str
    turns: int
    order_id: str
    refund_amount: float
    approval_note: str
    ticket: dict


# ============================================================
# 3. 提示词与解析（与 V2 相同，保证语义一致）
# ============================================================
SUPERVISOR_PROMPT = """你是电商客服系统的总控路由，负责把「用户当前这句话」分派给合适的专员。

可选目标（只输出其中一个词，不要输出任何解释）：
- product_advisor：售前咨询，包括商品参数、价格、库存、优惠券、是否支持退换等
- after_sales：售后处理，包括查订单、查物流、申请退款、退换货政策纠纷
- ticket：用户要求生成工单，或需要对本次会话做质检总结

规则：
1. 只依据【最后一条用户消息】判断；用户再次提问（如“那如果打 9 折呢？”）也是新诉求，需照常分派；
2. 你只负责从上述三者中选一个，不要输出 FINISH —— 本轮是否结束，由系统根据专员是否已给出最终答复自动判定。
"""

PRESALE_PROMPT = """你是电商平台的售前导购专员，负责商品咨询。

你可以使用：商品知识检索、商品详情查询、计算三类工具。
你的职责边界：只回答商品与价格相关问题（参数、库存、价格、优惠、退换条件）。
你不具备查询订单和执行退款的能力，遇到售后诉求请如实说明需转售后专员处理。
回答必须基于工具返回内容，不得编造商品信息或优惠政策。
"""

AFTERSALE_PROMPT = """你是电商平台的售后处理专员，负责订单与售后问题。

你可以使用：售后政策检索、订单查询、物流查询、执行退款、计算五类工具。

工作流程要求：
1. 涉及具体订单时，先调用 get_order_status 确认订单信息与实付金额，不要凭空假设；
2. 涉及退货/退款规则时，先用知识检索确认政策；
3. 确认可以退款后，调用 apply_refund 提交退款申请（金额不得超过订单实付金额）；
4. 大额退款会由系统转人工审批，你无需自行判断金额是否超限。

回答必须基于工具返回内容，不得编造订单信息或售后政策。
"""

TICKET_PROMPT = """你是客服质检与工单专员。请把本次客服会话总结成一张结构化售后工单。

只输出 JSON，不要输出任何解释文字或代码块标记。字段如下：
{
  "category": "售前 | 售后 | 投诉",
  "summary": "一句话概述用户诉求与处理结果",
  "order_id": "涉及的订单号，没有则空字符串",
  "resolution": "最终处理结果，没有则空字符串",
  "need_follow_up": true 或 false
}
"""

_ROUTES = ("product_advisor", "after_sales", "ticket", "FINISH")


def _parse_route(text: str) -> str:
    lowered = (text or "").lower()
    for token in _ROUTES:
        if token.lower() in lowered:
            return token
    return "FINISH"


def _fallback_route(message) -> str:
    """模型未给出有效路由（含误输出 FINISH）时的确定性兜底。

    进入此处说明「最后一条是用户的新发言」，本轮必须分派给某位专员，绝不能空结束。
    """
    text = message.content if isinstance(message.content, str) else str(message.content)
    if any(k in text for k in ("订单", "物流", "退款", "退货", "换货", "售后", "发票")):
        return "after_sales"
    if any(k in text for k in ("工单", "质检", "总结", "投诉")):
        return "ticket"
    return "product_advisor"


def _parse_ticket(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {
        "category": "未知",
        "summary": text[:120],
        "order_id": "",
        "resolution": "",
        "need_follow_up": False,
    }


# ============================================================
# 4. 组装图：工具由 MCP 动态提供，节点全部异步
# ============================================================
def route_after_sales(state: AgentState) -> str:
    """售后专员产出工具调用后的分流：需审批 → approval；否则 → 工具节点；无调用 → 回 supervisor。"""
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        return "supervisor"
    if needs_human_approval(calls):
        return "approval"
    return "aftersale_tools"


def build_v3_graph(presale_tools, aftersale_tools, llm_client=None):
    """用 MCP 分组工具构建 V3 图（与 V2 同构，节点异步）。

    参数：
      presale_tools:   售前专员可用的工具（来自 MCP，已按白名单分组）
      aftersale_tools: 售后专员可用的工具
      llm_client:      可注入的 LLM（默认用 get_llm()，便于测试替换）
    """
    base_llm = llm_client or get_llm()
    presale_llm = base_llm.bind_tools(presale_tools)
    aftersale_llm = base_llm.bind_tools(aftersale_tools)

    async def supervisor(state: AgentState, *, store) -> Command:
        turns = int(state.get("turns") or 0) + 1
        if turns > MAX_SUPERVISOR_TURNS:
            return Command(goto=END, update={"intent": "finish", "turns": turns})

        last = state["messages"][-1]
        # 确定性结束：最后一条已是专员的最终答复（无工具调用）→ 本轮完成。
        # 不把「是否结束」交给 LLM 判断，避免多轮追问时被误判为 FINISH。
        if isinstance(last, AIMessage) and not (getattr(last, "tool_calls", None) or []):
            return Command(goto=END, update={"intent": "finish", "turns": turns})

        profile_hint = ""
        uid = state.get("user_id")
        if uid:
            item = store.get(("memory", uid), "profile")
            if item is not None:
                profile_hint = f"\n【已知用户画像】{json.dumps(item.value, ensure_ascii=False)}"

        response = await base_llm.ainvoke(
            [SystemMessage(content=SUPERVISOR_PROMPT + profile_hint)] + list(state["messages"])
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        route = _parse_route(content)

        # 用户刚发言，必须分派；模型若误输出 FINISH/无法识别，用关键词确定性兜底。
        if route == "FINISH":
            route = _fallback_route(last)
        return Command(goto=route, update={"intent": route, "turns": turns})

    async def product_advisor(state: AgentState) -> dict:
        response = await presale_llm.ainvoke(
            [SystemMessage(content=PRESALE_PROMPT)] + list(state["messages"])
        )
        return {"messages": [response]}

    async def after_sales(state: AgentState) -> dict:
        response = await aftersale_llm.ainvoke(
            [SystemMessage(content=AFTERSALE_PROMPT)] + list(state["messages"])
        )
        return {"messages": [response]}

    def approval(state: AgentState) -> Command:
        """退款金额超阈值时暂停，等待人工决策（批准 → 执行；驳回 → 回售后专员）。"""
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        order_id, amount = extract_refund_request(calls)

        decision = interrupt(
            {
                "type": "refund_approval",
                "question": f"订单 {order_id} 申请退款 {amount} 元，达到审批阈值 {REFUND_APPROVAL_THRESHOLD} 元，需人工审批。",
                "order_id": order_id,
                "amount": amount,
                "options": ["approve", "reject"],
            }
        )

        approved = str(decision).strip().lower() in {"approve", "yes", "y", "批准", "同意", "通过"}
        base_update = {"order_id": order_id, "refund_amount": amount}

        if approved:
            return Command(
                goto="aftersale_tools",
                update={**base_update, "approval_note": f"人工审批通过（{amount} 元），已放行退款"},
            )

        rejected = [
            ToolMessage(
                content="人工审批未通过：该退款申请已被驳回，未执行退款。",
                tool_call_id=call["id"],
            )
            for call in calls
        ]
        return Command(
            goto="after_sales",
            update={
                **base_update,
                "messages": rejected,
                "approval_note": f"人工审批驳回（{amount} 元），退款未执行",
            },
        )

    async def ticket_agent(state: AgentState, *, store) -> dict:
        response = await base_llm.ainvoke(
            [SystemMessage(content=TICKET_PROMPT)] + list(state["messages"])
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        ticket = _parse_ticket(content)

        uid = state.get("user_id")
        if uid:
            store.put(("memory", uid), "last_ticket", ticket)

        summary = f"工单已生成｜类别：{ticket.get('category', '未知')}｜{ticket.get('summary', '')}"
        return {"messages": [AIMessage(content=summary)], "ticket": ticket}

    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor)
    builder.add_node("product_advisor", product_advisor)
    builder.add_node("after_sales", after_sales)
    builder.add_node("ticket", ticket_agent)
    builder.add_node("presale_tools", ToolNode(presale_tools))
    builder.add_node("aftersale_tools", ToolNode(aftersale_tools))
    builder.add_node("approval", approval)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "product_advisor",
        tools_condition,
        {"tools": "presale_tools", END: "supervisor"},
    )
    builder.add_edge("presale_tools", "product_advisor")

    builder.add_conditional_edges(
        "after_sales",
        route_after_sales,
        {
            "approval": "approval",
            "aftersale_tools": "aftersale_tools",
            "supervisor": "supervisor",
        },
    )
    builder.add_edge("aftersale_tools", "after_sales")

    builder.add_edge("ticket", "supervisor")

    return builder


# ============================================================
# 5. 轨迹打印与异步演示
# ============================================================
def _brief(value, limit: int = 110) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _print_update(node: str, update) -> None:
    if not isinstance(update, dict):
        return

    if node == "supervisor":
        print(f"  [supervisor] 路由 → {update.get('intent')}")
    elif node in ("product_advisor", "after_sales"):
        messages = update.get("messages") or []
        if not messages:
            return
        message = messages[-1]
        calls = getattr(message, "tool_calls", None) or []
        if calls:
            rendered = ", ".join(f"{c['name']}({_brief(c['args'], 60)})" for c in calls)
            print(f"  [{node}] 请求调用工具：{rendered}")
        else:
            print(f"  [{node}] 回答：{_brief(getattr(message, 'content', ''))}")
    elif node == "ticket":
        messages = update.get("messages") or []
        text = getattr(messages[-1], "content", "") if messages else ""
        print(f"  [ticket] {_brief(text)}")
    elif node in ("presale_tools", "aftersale_tools"):
        for message in update.get("messages") or []:
            print(f"  [{node}] 工具返回：{_brief(getattr(message, 'content', ''))}")
    elif node == "approval":
        note = update.get("approval_note")
        if note:
            print(f"  [approval] {note}")


async def _consume(graph, stream_input, config) -> bool:
    """异步消费一次 stream，打印轨迹；返回是否遇到中断。"""
    interrupted = False
    async for event in graph.astream(stream_input, config=config, stream_mode="updates"):
        if "__interrupt__" in event:
            interrupted = True
            for item in event["__interrupt__"]:
                info = getattr(item, "value", item)
                print(f"  ⏸  触发人工审批：{_brief(info, 160)}")
            continue
        for node, update in event.items():
            _print_update(node, update)
    return interrupted


async def run_scenario(graph, title: str, question: str, user_id: str, thread_id: str, resume=None) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    payload = {
        "messages": [HumanMessage(content=question)],
        "user_id": user_id,
        "turns": 0,
    }

    print("=" * 70)
    print(title)
    print("=" * 70)
    print("用户：", question)

    interrupted = await _consume(graph, payload, config)
    if interrupted and resume:
        print(f"  ▶  模拟人工审批：{resume}")
        await _consume(graph, Command(resume=resume), config)
    print()


SCENARIOS = [
    ("1", "场景 1：售前咨询 → product_advisor", "星耀 X1 手机现在多少钱？如果打 9 折是多少？", "U001", "v3-presale", None),
    (
        "2",
        "场景 2：小额退款（59.9 元 < 阈值）→ 自动执行",
        "订单 SO20260902002 的有机纯牛奶到货就变质了，我在签收 48 小时内已拍好开箱照片，符合退款条件，请直接帮我退款 59.9 元。",
        "U002",
        "v3-refund-small",
        None,
    ),
    (
        "3",
        "场景 3：大额退款（2899 元 ≥ 阈值）→ 人工批准",
        "订单 SO20260905005 的清风 K3 变频空调出现质量故障，我已在签收 15 天内并拍好故障视频，符合退货退款条件，请直接为我办理全额退款 2899 元。",
        "U004",
        "v3-refund-approve",
        "yes",
    ),
    (
        "4",
        "场景 4：大额退款（599 元 ≥ 阈值）→ 人工驳回",
        "订单 SO20260904004 的声动 P1 耳机出现故障，我已提供故障视频，符合退货条件，请直接帮我退款 599 元。",
        "U001",
        "v3-refund-reject",
        "no",
    ),
]


async def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None

    # 演示可重复：退款会写回订单文件，先重置为初始状态
    count = order_store.reset_orders()
    print(f"[v3] 已重置演示订单数据（{count} 条）\n")

    store = InMemoryStore()
    store.put(
        ("memory", "U001"),
        "profile",
        {"name": "张先生", "level": "黄金会员", "prefer": ["数码", "家电"]},
    )

    # MCP 工具与会话绑定，整段演示必须在 async with 作用域内
    async with MCPToolHub() as hub:
        graph = build_v3_graph(hub.presale_tools, hub.aftersale_tools).compile(
            checkpointer=MemorySaver(), store=store
        )

        for key, title, question, user_id, thread_id, resume in SCENARIOS:
            if only and only != key:
                continue
            await run_scenario(graph, title, question, user_id, thread_id, resume)


if __name__ == "__main__":
    asyncio.run(main())
