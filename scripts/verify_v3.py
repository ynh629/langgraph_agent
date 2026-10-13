"""
scripts/verify_v3.py
====================
V3 回归验证（真实 LLM）：端到端跑通 4 个场景，用【具体断言】对照 V2 的语义。

断言标准（写死，不依赖人工目测）：
  场景 1 售前咨询：最终回复同时包含商品名（星耀 X1）与价格（3999 或折后 3599）
  场景 2 小额退款：apply_refund 被调用，且订单文件状态变为「已退款」（refund_status=退款中）
  场景 3 大额批准：触发 interrupt；resume="yes" 后继续执行，订单状态变为「已退款」
  场景 4 大额驳回：触发 interrupt；resume="no" 后返回驳回消息，订单状态保持未退款

关于稳定性（重要）：
  真实 LLM 即便 temperature=0 也可能对同一句话给出不同决策（例如像真人客服一样先追问
  凭证，而非直接提交退款），因此本脚本的结果会有波动。硬门禁请用配套的
  scripts/verify_v3_deterministic.py（注入脚本化 LLM，断言可稳定复现）。

说明：
  - 断言基于「图最终状态 + 订单文件落盘」这类确定性产物，而不是 LLM 自由文本的措辞；
  - 订单的「已退款」在数据层体现为 refund_status = "退款中"（见 order_store.execute_refund）。
  - 场景 2/3 涉及语义检索时会构建 Chroma 索引并写库，受限沙箱可能拦截该写入；
    若命中该情况会明确提示在本机终端重跑，而非静默通过。

运行：
    python scripts/verify_v3.py          # 全部 4 个场景
    python scripts/verify_v3.py 2        # 只跑第 2 个场景
"""
import asyncio
import os
import re
import sys

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_AGENT_DIR, os.path.join(_AGENT_DIR, "检索")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

import order_store
from mcp_client import MCPToolHub
from v3_agent import build_v3_graph

# 应用内「已退款」状态值（order_store.execute_refund 写入的中文状态）
REFUNDED_STATUS = "退款中"

# 各场景使用的订单（均取「已完成」状态，避免因订单在途导致售后专员拒绝受理）
SMALL_ORDER, SMALL_AMOUNT = "SO20260902002", 59.9
APPROVE_ORDER, APPROVE_AMOUNT = "SO20260905005", 2899.0
REJECT_ORDER, REJECT_AMOUNT = "SO20260904004", 599.0

# 场景定义：(key, 标题, 用户问题, user_id, thread_id, resume)
# 说明：回归场景刻意写得「明确且可直接执行」（给出订单号 + 明确金额 + 满足政策前提），
#       否则 LLM 会像真人客服一样先追问证据，导致断言不可复现（V2 同样如此）。
CASES = [
    ("1", "场景 1：售前咨询", "星耀 X1 手机现在多少钱？如果打 9 折是多少？", "U001", "v3v-presale", None),
    (
        "2",
        "场景 2：小额退款（自动执行）",
        f"订单 {SMALL_ORDER} 的有机纯牛奶到货就变质了，我在签收 48 小时内已拍好开箱照片，符合退款条件，请直接帮我退款 {SMALL_AMOUNT} 元。",
        "U002",
        "v3v-refund-small",
        None,
    ),
    (
        "3",
        "场景 3：大额退款（人工批准）",
        f"订单 {APPROVE_ORDER} 的清风 K3 变频空调出现质量故障，我已在签收 15 天内并拍好故障视频，符合退货退款条件，请直接为我办理全额退款 {APPROVE_AMOUNT} 元。",
        "U004",
        "v3v-refund-approve",
        "yes",
    ),
    (
        "4",
        "场景 4：大额退款（人工驳回）",
        f"订单 {REJECT_ORDER} 的声动 P1 耳机出现故障，我已提供故障视频，符合退货条件，请直接帮我退款 {REJECT_AMOUNT} 元。",
        "U001",
        "v3v-refund-reject",
        "no",
    ),
]


async def _drive(graph, stream_input, config) -> tuple:
    """驱动一次 astream，返回 (是否中断, 中断载荷)。"""
    interrupted = False
    interrupt_value = None
    async for event in graph.astream(stream_input, config=config, stream_mode="updates"):
        if "__interrupt__" in event:
            interrupted = True
            for item in event["__interrupt__"]:
                interrupt_value = getattr(item, "value", item)
    return interrupted, interrupt_value


async def run_case(graph, question: str, user_id: str, thread_id: str, resume=None) -> tuple:
    """跑一个场景，返回 (是否中断, 中断载荷, 最终状态)。"""
    config = {"configurable": {"thread_id": thread_id}}
    payload = {"messages": [HumanMessage(content=question)], "user_id": user_id, "turns": 0}

    interrupted, interrupt_value = await _drive(graph, payload, config)
    if interrupted and resume is not None:
        await _drive(graph, Command(resume=resume), config)

    snapshot = await graph.aget_state(config)
    return interrupted, interrupt_value, snapshot.values


def _all_text(messages) -> str:
    parts = []
    for m in messages:
        content = getattr(m, "content", "")
        parts.append(content if isinstance(content, str) else str(content))
    return "\n".join(parts)


def _called_tools(messages) -> list:
    """收集本轮实际被调用的工具名（来自 AI 消息的 tool_calls）。"""
    names = []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            names.append(call.get("name"))
    return names


# ------------------------------------------------------------
# 各场景断言
# ------------------------------------------------------------
def assert_presale(values) -> None:
    text = _all_text(values["messages"])
    assert "星耀" in text, f"最终回复未包含商品名：{text[-300:]}"
    assert re.search(r"3999|3599", text), f"最终回复未包含价格：{text[-300:]}"
    assert "apply_refund" not in _called_tools(values["messages"]), "售前场景不应调用退款工具"
    print("  [PASS] 售前：回复含商品名与价格，且未越权调用退款")


def assert_small_refund(values) -> None:
    assert "apply_refund" in _called_tools(values["messages"]), "小额退款未调用 apply_refund"
    order = order_store.get_order(SMALL_ORDER)
    assert order is not None, f"订单 {SMALL_ORDER} 不存在"
    assert order.get("refund_status") == REFUNDED_STATUS, f"订单状态未变为已退款：{order.get('refund_status')}"
    assert abs(float(order.get("refund_amount")) - SMALL_AMOUNT) < 0.01, order.get("refund_amount")
    print("  [PASS] 小额退款：apply_refund 已调用，订单落盘状态 =", order.get("refund_status"))


def assert_approve(values, interrupted, interrupt_value) -> None:
    assert interrupted, "大额退款未触发 interrupt 审批"
    assert (interrupt_value or {}).get("type") == "refund_approval", interrupt_value
    note = values.get("approval_note") or ""
    assert "通过" in note, f"审批结论异常：{note}"
    order = order_store.get_order(APPROVE_ORDER)
    assert order.get("refund_status") == REFUNDED_STATUS, f"批准后订单未退款：{order.get('refund_status')}"
    assert abs(float(order.get("refund_amount")) - APPROVE_AMOUNT) < 0.01, order.get("refund_amount")
    print("  [PASS] 大额批准：interrupt 触发，resume 后订单落盘状态 =", order.get("refund_status"))


def assert_reject(values, interrupted, interrupt_value) -> None:
    assert interrupted, "大额退款未触发 interrupt 审批"
    assert (interrupt_value or {}).get("type") == "refund_approval", interrupt_value
    note = values.get("approval_note") or ""
    assert "驳回" in note, f"驳回结论异常：{note}"
    text = _all_text(values["messages"])
    assert "驳回" in text or "未执行退款" in text, f"未返回驳回消息：{text[-300:]}"
    order = order_store.get_order(REJECT_ORDER)
    assert not order.get("refund_status"), f"驳回后订单不应有退款记录：{order.get('refund_status')}"
    print("  [PASS] 大额驳回：interrupt 触发，返回驳回消息，订单保持未退款")


CHECKERS = {
    "1": lambda v, i, iv: assert_presale(v),
    "2": lambda v, i, iv: assert_small_refund(v),
    "3": lambda v, i, iv: assert_approve(v, i, iv),
    "4": lambda v, i, iv: assert_reject(v, i, iv),
}


async def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None

    count = order_store.reset_orders()
    print(f"[verify_v3] 已重置订单数据（{count} 条）\n")

    store = InMemoryStore()
    store.put(
        ("memory", "U001"),
        "profile",
        {"name": "张先生", "level": "黄金会员", "prefer": ["数码", "家电"]},
    )

    passed = []
    async with MCPToolHub() as hub:
        graph = build_v3_graph(hub.presale_tools, hub.aftersale_tools).compile(
            checkpointer=MemorySaver(), store=store
        )

        for key, title, question, user_id, thread_id, resume in CASES:
            if only and only != key:
                continue
            print(f"[{key}] {title}")
            interrupted, interrupt_value, values = await run_case(
                graph, question, user_id, thread_id, resume
            )
            CHECKERS[key](values, interrupted, interrupt_value)
            passed.append(key)
            print()

    print(f"[verify_v3] V3 REGRESSION PASS：场景 {', '.join(passed)}")


if __name__ == "__main__":
    asyncio.run(main())
