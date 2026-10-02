"""
scripts/verify_v3_deterministic.py
==================================
V3 确定性回归：用「脚本化 LLM」驱动 V3 图，把图的编排正确性（路由、MCP 工具执行、
审批阈值、interrupt/resume、订单落盘）与真实 LLM 的决策随机性解耦，使 4 个场景的
断言可稳定复现。

为什么需要它（面试可讲）：
  真实 LLM 即便 temperature=0，也可能对同一句话给出不同决策（有时像真人客服先追问
  凭证，而不是直接提交退款）。把硬断言压在真实 LLM 上，回归会时好时坏。因此：
    - 本脚本：注入假 LLM，确定性验证「图 + MCP 工具链」是否正确 —— 作为硬门禁；
    - verify_v3.py：真实 LLM 端到端回归 —— 验证「端到端可用」，结果受 LLM 波动影响。

断言标准与 verify_v3.py 完全一致（直接复用其断言函数与场景定义，避免两处标准漂移）。

运行：
    python scripts/verify_v3_deterministic.py
    python scripts/verify_v3_deterministic.py 2
"""
import asyncio
import os
import sys

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_AGENT_DIR, os.path.join(_AGENT_DIR, "检索")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

import order_store
from mcp_client import AFTERSALE_TOOL_NAMES, PRESALE_TOOL_NAMES, MCPToolHub
from v3_agent import build_v3_graph

# 复用 verify_v3 的场景定义与断言，保证两套验证口径一致
from verify_v3 import (
    APPROVE_AMOUNT,
    APPROVE_ORDER,
    CASES,
    CHECKERS,
    REJECT_AMOUNT,
    REJECT_ORDER,
    SMALL_AMOUNT,
    SMALL_ORDER,
    run_case,
)


# ============================================================
# 脚本化 LLM：按固定剧本返回消息，让图的编排可确定性验证
# ============================================================
class ScriptedLLM:
    """按剧本顺序返回 AIMessage 的假 LLM。

    - bind_tools 返回自身，并记录每次绑定的工具名（用于校验工具权限隔离）；
    - ainvoke 依次弹出剧本；剧本耗尽即报错（说明图执行次数与预期不符）。
    """

    def __init__(self, script):
        self._script = list(script)
        self._index = 0
        self.bound_sets = []

    def bind_tools(self, tools):
        self.bound_sets.append(sorted(getattr(t, "name", "") for t in tools))
        return self

    async def ainvoke(self, messages):
        if self._index >= len(self._script):
            raise AssertionError("ScriptedLLM 剧本耗尽：图执行次数超出预期")
        message = self._script[self._index]
        self._index += 1
        return message


def _route(target: str) -> AIMessage:
    """supervisor 的路由输出。"""
    return AIMessage(content=target)


def _refund(order_id: str, amount: float) -> AIMessage:
    """售后专员发出的 apply_refund 工具调用。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": "apply_refund", "args": {"order_id": order_id, "amount": amount}, "id": "call_refund"}],
    )


def _presale_lookup() -> AIMessage:
    """售前专员发出的「查详情 + 算折扣」工具调用。"""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "get_product_detail", "args": {"product_name": "星耀 X1 智能手机"}, "id": "call_detail"},
            {"name": "calculate", "args": {"expression": "3999 * 0.9"}, "id": "call_calc"},
        ],
    )


def _reply(text: str) -> AIMessage:
    return AIMessage(content=text)


def _scripts() -> dict:
    """各场景剧本：顺序 = 图中 LLM 节点的调用顺序。"""
    return {
        # supervisor → product_advisor → 工具 → product_advisor(终答) → supervisor(FINISH)
        "1": ScriptedLLM(
            [
                _route("product_advisor"),
                _presale_lookup(),
                _reply("星耀 X1 智能手机当前售价 3999 元，打 9 折后为 3599.1 元。"),
                _route("FINISH"),
            ]
        ),
        # supervisor → after_sales → 退款工具(小额放行) → after_sales(终答) → supervisor(FINISH)
        "2": ScriptedLLM(
            [
                _route("after_sales"),
                _refund(SMALL_ORDER, SMALL_AMOUNT),
                _reply("已为您提交 59.9 元退款申请。"),
                _route("FINISH"),
            ]
        ),
        # supervisor → after_sales → approval(interrupt) → resume=yes → 退款工具 → after_sales(终答) → supervisor(FINISH)
        "3": ScriptedLLM(
            [
                _route("after_sales"),
                _refund(APPROVE_ORDER, APPROVE_AMOUNT),
                _reply("人工审批已通过，退款已提交。"),
                _route("FINISH"),
            ]
        ),
        # supervisor → after_sales → approval(interrupt) → resume=no → after_sales(终答) → supervisor(FINISH)
        "4": ScriptedLLM(
            [
                _route("after_sales"),
                _refund(REJECT_ORDER, REJECT_AMOUNT),
                _reply("该退款申请已被驳回，未执行退款。"),
                _route("FINISH"),
            ]
        ),
    }


def assert_bindings(llm: ScriptedLLM) -> None:
    """校验两个专员绑定的工具集与白名单完全一致（工具权限隔离）。"""
    expected = [sorted(PRESALE_TOOL_NAMES), sorted(AFTERSALE_TOOL_NAMES)]
    assert llm.bound_sets == expected, f"工具绑定与白名单不符：{llm.bound_sets}"
    assert "apply_refund" not in llm.bound_sets[0], "售前专员不应绑定退款工具"
    assert "get_order_status" not in llm.bound_sets[0], "售前专员不应绑定订单工具"
    print("  [PASS] 工具绑定：售前/售后两组与白名单一致，售前组无订单/退款工具")


async def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None

    count = order_store.reset_orders()
    print(f"[verify_v3_det] 已重置订单数据（{count} 条）\n")

    store = InMemoryStore()
    store.put(
        ("memory", "U001"),
        "profile",
        {"name": "张先生", "level": "黄金会员", "prefer": ["数码", "家电"]},
    )

    scripts = _scripts()
    passed = []

    async with MCPToolHub() as hub:
        for key, title, question, user_id, thread_id, resume in CASES:
            if only and only != key:
                continue
            print(f"[{key}] {title}（脚本化 LLM）")
            scripted = scripts[key]
            graph = build_v3_graph(hub.presale_tools, hub.aftersale_tools, llm_client=scripted).compile(
                checkpointer=MemorySaver(), store=store
            )
            assert_bindings(scripted)

            interrupted, interrupt_value, values = await run_case(
                graph, question, user_id, f"det-{thread_id}", resume
            )
            if key == "2":
                assert not interrupted, "小额退款不应触发人工审批"
            CHECKERS[key](values, interrupted, interrupt_value)
            passed.append(key)
            print()

    print(f"[verify_v3_det] V3 DETERMINISTIC REGRESSION PASS：场景 {', '.join(passed)}")


if __name__ == "__main__":
    asyncio.run(main())
