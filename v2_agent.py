"""
v2_agent.py
===========
V2：电商客服多智能体系统（Supervisor + 三专员 + 人工审批 + 长期记忆）

在 V1「单 agent + 三工具」基础上升级为多智能体协同：

  1. Supervisor 总控路由：意图识别后把请求分派给对应专员（Command handoff）
  2. 三个专员：
       product_advisor  售前（商品参数 / 价格 / 优惠）
       after_sales      售后（订单 / 物流 / 政策 / 退款）
       ticket           质检与工单生成
  3. 人工审批：退款金额 >= REFUND_APPROVAL_THRESHOLD 时 interrupt 暂停，等待人工批准或驳回
  4. 长期记忆：store 跨会话保存用户画像与最近工单

关键设计（面试可讲）：
  - 路由用 Command(goto=...) 而非 conditional edges：一个节点同时完成「更新状态 + 指定下一跳」
  - 工具权限隔离：售前专员只绑定商品类工具，其工具节点里根本没有退款工具，结构上不可能发起退款
  - 审批写在 approval 节点，apply_refund 工具保持纯执行，两者解耦、工具可单测

运行：
    python v2_agent.py          # 依次跑 4 个演示场景
    python v2_agent.py 2        # 只跑第 2 个场景

重置演示数据（退款会写回订单文件）：
    python order_store.py --reset
"""
import json
import os
import sys

# 离线优先：本地已缓存 BGE 模型（~/.cache/huggingface/local），跳过 HuggingFace 联网检查，
# 避免 WinError 10060 连接超时触发的 5 次重试阻塞。必须在导入检索链之前设置。
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

from tools import AFTERSALE_TOOLS, PRESALE_TOOLS
from policy import (
    REFUND_APPROVAL_THRESHOLD,
    extract_refund_request,
    needs_human_approval,
)

load_dotenv()

# 路由次数上限，防止 supervisor 与专员之间反复横跳导致死循环
MAX_SUPERVISOR_TURNS = 8


# ============================================================
# 1. LLM
# ============================================================
llm = ChatOpenAI(
    model_name=os.getenv("LLM_MODEL", "deepseek-flash"),
    temperature=0,
    max_tokens=1500,
    api_key=os.getenv("deepseek_api_key"),
    base_url=os.getenv("LLM_BASE_URL"),
)

presale_llm = llm.bind_tools(PRESALE_TOOLS)
aftersale_llm = llm.bind_tools(AFTERSALE_TOOLS)


# ============================================================
# 2. 共享 State
#    messages 来自 MessagesState（带 add_messages reducer，追加+去重）；
#    其余业务字段无 reducer，返回时按「覆盖」处理。
# ============================================================
class AgentState(MessagesState):
    user_id: str            # 用户标识，用于 store 长期记忆
    intent: str             # supervisor 本轮路由结果
    turns: int              # 路由轮次，用于防死循环
    order_id: str           # 售后场景涉及的订单号
    refund_amount: float    # 售后场景的退款金额
    approval_note: str      # 人工审批结论
    ticket: dict            # 质检生成的工单


# ============================================================
# 3. Supervisor：意图识别 + 路由（Command handoff）
# ============================================================
SUPERVISOR_PROMPT = """你是电商客服系统的总控路由，负责把用户请求分派给合适的专员。

可选目标（只输出其中一个词，不要输出解释）：
- product_advisor：售前咨询，包括商品参数、价格、库存、优惠券、是否支持退换等
- after_sales：售后处理，包括查订单、查物流、申请退款、退换货政策纠纷
- ticket：用户要求生成工单，或需要对本次会话做质检总结
- FINISH：用户问题已被完整回答，无需再分派

判断依据是【最后一条用户消息】。若最近已有专员给出完整答复，输出 FINISH。
"""

_ROUTES = ("product_advisor", "after_sales", "ticket", "FINISH")


def _parse_route(text: str) -> str:
    lowered = (text or "").lower()
    for token in _ROUTES:
        if token.lower() in lowered:
            return token
    return "FINISH"


def supervisor(state: AgentState, *, store) -> Command:
    turns = int(state.get("turns") or 0) + 1
    if turns > MAX_SUPERVISOR_TURNS:
        return Command(goto=END, update={"intent": "finish", "turns": turns})

    # 长期记忆：读取用户画像，注入路由上下文
    profile_hint = ""
    uid = state.get("user_id")
    if uid:
        item = store.get(("memory", uid), "profile")
        if item is not None:
            profile_hint = f"\n【已知用户画像】{json.dumps(item.value, ensure_ascii=False)}"

    response = llm.invoke(
        [SystemMessage(content=SUPERVISOR_PROMPT + profile_hint)] + list(state["messages"])
    )
    content = response.content if isinstance(response.content, str) else str(response.content)
    route = _parse_route(content)

    if route == "FINISH":
        return Command(goto=END, update={"intent": "finish", "turns": turns})
    return Command(goto=route, update={"intent": route, "turns": turns})


# ============================================================
# 4. 售前专员
# ============================================================
PRESALE_PROMPT = """你是电商平台的售前导购专员，负责商品咨询。

你可以使用：商品知识检索、商品详情查询、计算三类工具。
你的职责边界：只回答商品与价格相关问题（参数、库存、价格、优惠、退换条件）。
你不具备查询订单和执行退款的能力，遇到售后诉求请如实说明需转售后专员处理。
回答必须基于工具返回内容，不得编造商品信息或优惠政策。
"""


def product_advisor(state: AgentState):
    response = presale_llm.invoke(
        [SystemMessage(content=PRESALE_PROMPT)] + list(state["messages"])
    )
    return {"messages": [response]}


# ============================================================
# 5. 售后专员
# ============================================================
AFTERSALE_PROMPT = """你是电商平台的售后处理专员，负责订单与售后问题。

你可以使用：售后政策检索、订单查询、物流查询、执行退款、计算五类工具。

工作流程要求：
1. 涉及具体订单时，先调用 get_order_status 确认订单信息与实付金额，不要凭空假设；
2. 涉及退货/退款规则时，先用知识检索确认政策；
3. 确认可以退款后，调用 apply_refund 提交退款申请（金额不得超过订单实付金额）；
4. 大额退款会由系统转人工审批，你无需自行判断金额是否超限。

回答必须基于工具返回内容，不得编造订单信息或售后政策。
"""


def after_sales(state: AgentState):
    response = aftersale_llm.invoke(
        [SystemMessage(content=AFTERSALE_PROMPT)] + list(state["messages"])
    )
    return {"messages": [response]}


def route_after_sales(state: AgentState) -> str:
    """售后专员产出工具调用后的分流：需审批 → approval；否则 → 工具节点；无调用 → 回 supervisor。"""
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        return "supervisor"
    if needs_human_approval(calls):
        return "approval"
    return "aftersale_tools"


# ============================================================
# 6. 人工审批节点（interrupt / resume）
# ============================================================
def approval(state: AgentState) -> Command:
    """
    退款金额超阈值时暂停，等待人工决策。
    批准 → 进入售后工具节点执行退款；驳回 → 直接回售后专员并注入驳回结果。
    """
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


# ============================================================
# 7. 质检 / 工单专员
# ============================================================
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


def ticket_agent(state: AgentState, *, store) -> dict:
    response = llm.invoke(
        [SystemMessage(content=TICKET_PROMPT)] + list(state["messages"])
    )
    content = response.content if isinstance(response.content, str) else str(response.content)
    ticket = _parse_ticket(content)

    # 长期记忆：把最近一次工单写入 store，供后续会话参考
    uid = state.get("user_id")
    if uid:
        store.put(("memory", uid), "last_ticket", ticket)

    summary = f"工单已生成｜类别：{ticket.get('category', '未知')}｜{ticket.get('summary', '')}"
    return {"messages": [AIMessage(content=summary)], "ticket": ticket}


# ============================================================
# 8. 组装图
#    工具节点按角色拆成两个，实现工具权限隔离
# ============================================================
def build_v2_graph():
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor)
    builder.add_node("product_advisor", product_advisor)
    builder.add_node("after_sales", after_sales)
    builder.add_node("ticket", ticket_agent)
    builder.add_node("presale_tools", ToolNode(PRESALE_TOOLS))
    builder.add_node("aftersale_tools", ToolNode(AFTERSALE_TOOLS))
    builder.add_node("approval", approval)

    builder.add_edge(START, "supervisor")

    # supervisor 通过 Command(goto=...) 决定去向，无需显式条件边
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
# 9. 轨迹打印与演示
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


def _consume(graph, stream_input, config) -> bool:
    """消费一次 stream，打印轨迹；返回是否遇到中断。"""
    interrupted = False
    for event in graph.stream(stream_input, config=config, stream_mode="updates"):
        if "__interrupt__" in event:
            interrupted = True
            for item in event["__interrupt__"]:
                info = getattr(item, "value", item)
                print(f"  ⏸  触发人工审批：{_brief(info, 160)}")
            continue
        for node, update in event.items():
            _print_update(node, update)
    return interrupted


def run_scenario(graph, title: str, question: str, user_id: str, thread_id: str, resume=None) -> None:
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

    interrupted = _consume(graph, payload, config)
    if interrupted and resume:
        print(f"  ▶  模拟人工审批：{resume}")
        _consume(graph, Command(resume=resume), config)
    print()


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None

    store = InMemoryStore()
    # 预置用户画像，演示 store 长期记忆的读取
    store.put(
        ("memory", "U001"),
        "profile",
        {"name": "张先生", "level": "黄金会员", "prefer": ["数码", "家电"]},
    )

    graph = build_v2_graph().compile(checkpointer=MemorySaver(), store=store)

    scenarios = [
        ("1", "场景 1：售前咨询 → product_advisor", "星耀 X1 手机现在多少钱？如果打 9 折是多少？", "U001", "v2-presale", None),
        ("2", "场景 2：小额退款（59.9 元 < 阈值）→ 自动执行", "我的订单 SO20260902002 牛奶到货就变质了，帮我退款", "U002", "v2-refund-small", None),
        ("3", "场景 3：大额退款（3999 元 ≥ 阈值）→ 人工批准", "订单 SO20260901001 的手机有质量问题，我要全额退款 3999 元", "U001", "v2-refund-approve", "approve"),
        ("4", "场景 4：大额退款（599 元 ≥ 阈值）→ 人工驳回", "订单 SO20260904004 的耳机我不想要了，退款 599 元", "U001", "v2-refund-reject", "reject"),
    ]

    for key, title, question, user_id, thread_id, resume in scenarios:
        if only and only != key:
            continue
        run_scenario(graph, title, question, user_id, thread_id, resume)


if __name__ == "__main__":
    main()
