"""
scripts/verify_mcp_servers.py
=============================
P1/P2 验证：以 stdio 同时拉起「检索」「订单」两个 MCP Server，
验证工具发现、名称/描述、以及实际调用。

注意：search_product_knowledge 首次调用会构建 Chroma 索引并写入 chroma_db，
      若在受限沙箱中运行可能被拦截；该步骤失败会单独提示，不算整体失败。

运行：
    python scripts/verify_mcp_servers.py
"""
import asyncio
import json
import os
import sys

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_AGENT_DIR, os.path.join(_AGENT_DIR, "检索")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

_EXPECTED = sorted(
    [
        "search_product_knowledge",
        "get_product_detail",
        "calculate",
        "get_order_status",
        "get_logistics",
        "apply_refund",
    ]
)


def _server(name: str, script: str) -> dict:
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-u", os.path.join(_AGENT_DIR, "mcp_servers", script)],
        "cwd": _AGENT_DIR,
        "encoding": "utf-8",
        "encoding_error_handler": "strict",
        "env": {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    }


def build_config() -> dict:
    return {
        "retrieval": _server("retrieval", "retrieval_server.py"),
        "order": _server("order", "order_server.py"),
    }


def _extract_text(result) -> str:
    """MCP 工具结果经适配器转换后形如 [{'type':'text','text':...}]，取出正文。"""
    if isinstance(result, list) and result and isinstance(result[0], dict) and "text" in result[0]:
        return result[0]["text"]
    return str(result)


async def main() -> None:
    client = MultiServerMCPClient(build_config())

    async with client.session("retrieval") as retrieval_session, client.session("order") as order_session:
        retrieval_tools = await load_mcp_tools(retrieval_session)
        order_tools = await load_mcp_tools(order_session)
        tools = {t.name: t for t in retrieval_tools + order_tools}

        names = sorted(tools)
        print("[verify] 检索服务工具：", sorted(t.name for t in retrieval_tools))
        print("[verify] 订单服务工具：", sorted(t.name for t in order_tools))
        assert names == _EXPECTED, f"工具集合不符：{names}"

        # 描述必须非空（LLM 靠它决定何时调用）
        empty = [n for n, t in tools.items() if not (t.description or "").strip()]
        assert not empty, f"以下工具缺少描述：{empty}"
        print("[verify] 工具描述齐全（非空）")

        calc = await tools["calculate"].ainvoke({"expression": "3999 * 0.9"})
        print("[verify] calculate =", calc)
        assert "3599" in str(calc), calc

        product = await tools["get_product_detail"].ainvoke({"product_name": "星耀 X1 智能手机"})
        print("[verify] get_product_detail =", str(product)[:80])
        assert "星耀 X1" in str(product), product

        order = await tools["get_order_status"].ainvoke({"order_id": "SO20260902002"})
        print("[verify] get_order_status =", str(order)[:80])
        assert "SO20260902002" in str(order), order

        # 语义检索：依赖 chroma 写权限。
        # 这一步同时是「stdout 协议流是否干净」的检验：若服务端日志漏进 stdout，
        # 适配器解析会失败，返回的就不是可解析的结果 JSON。
        search_result = await tools["search_product_knowledge"].ainvoke(
            {"query": "生鲜牛奶变质 退款政策", "top_k": 3}
        )
        payload = _extract_text(search_result)
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            if "chroma" in payload.lower() or "database" in payload.lower():
                print("[verify] search_product_knowledge 跳过：chroma 数据库不可写（沙箱限制），请在本机终端重跑")
            else:
                raise AssertionError(f"检索返回非结果内容（疑似协议流被污染）：{payload[:200]}")
        else:
            assert isinstance(parsed, list) and parsed and "title_path" in parsed[0], parsed
            print("[verify] search_product_knowledge 命中条数 =", len(parsed))

    print("[verify] MCP SERVERS PASS")


if __name__ == "__main__":
    asyncio.run(main())
