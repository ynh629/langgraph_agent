"""
scripts/verify_mcp_client.py
============================
P3 验证：MCPToolHub 能否聚合两个 Server，并按白名单把工具分成
售前 / 售后两组，且分组工具可正常调用。

运行：
    python scripts/verify_mcp_client.py
"""
import asyncio
import os
import sys

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_AGENT_DIR, os.path.join(_AGENT_DIR, "检索")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mcp_client import (
    AFTERSALE_TOOL_NAMES,
    ALL_TOOL_NAMES,
    PRESALE_TOOL_NAMES,
    MCPToolHub,
)


async def main() -> None:
    async with MCPToolHub() as hub:
        presale = sorted(t.name for t in hub.presale_tools)
        aftersale = sorted(t.name for t in hub.aftersale_tools)
        all_names = sorted(t.name for t in hub.all_tools)

        print("[verify] presale  :", presale)
        print("[verify] aftersale:", aftersale)

        assert presale == sorted(PRESALE_TOOL_NAMES), presale
        assert aftersale == sorted(AFTERSALE_TOOL_NAMES), aftersale
        assert all_names == sorted(ALL_TOOL_NAMES), all_names

        # 权限隔离：售前组不得出现订单 / 退款类工具
        assert "apply_refund" not in presale, "售前组混入了退款工具"
        assert "get_order_status" not in presale, "售前组混入了订单工具"
        print("[verify] 分组正确，且售前组无订单/退款工具（权限隔离 OK）")

        by_name = {t.name: t for t in hub.all_tools}
        calc = await by_name["calculate"].ainvoke({"expression": "(3999 * 0.9) - 200"})
        print("[verify] calculate =", calc)
        assert "3399.1" in str(calc), calc

        order = await by_name["get_order_status"].ainvoke({"order_id": "SO20260902002"})
        print("[verify] get_order_status =", str(order)[:80])
        assert "SO20260902002" in str(order), order

    print("[verify] MCP CLIENT PASS")


if __name__ == "__main__":
    asyncio.run(main())
