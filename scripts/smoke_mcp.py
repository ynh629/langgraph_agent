"""
scripts/smoke_mcp.py
====================
P0 兼容性冒烟：验证 langchain-mcp-adapters 能以 stdio 拉起 MCP Server、
动态发现工具、并成功调用（含中文参数与返回值）。

这是 V3 的前置门槛：若此脚本通过，说明「依赖版本 + stdio 传输 + 中文编码」
三条链路都可用，才值得继续做 P1 的检索服务化。

运行：
    python scripts/smoke_mcp.py
"""
import asyncio
import os
import sys

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_AGENT_DIR, os.path.join(_AGENT_DIR, "检索")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

_SERVER = os.path.join(_AGENT_DIR, "mcp_servers", "echo_server.py")


def build_config() -> dict:
    """stdio 连接配置：显式指定解释器、cwd 与 UTF-8，避免子进程编码踩坑。"""
    return {
        "echo": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-u", _SERVER],
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
    }


async def main() -> None:
    # 0.1.0+ 起 MultiServerMCPClient 不能直接 async with；
    # 持久会话必须用 client.session(server_name) + load_mcp_tools(session)。
    # 这一步对 V3 至关重要：会话常驻才能让检索服务的索引缓存在多次调用间存活。
    client = MultiServerMCPClient(build_config())
    async with client.session("echo") as session:
        tools = await load_mcp_tools(session)
        names = sorted(t.name for t in tools)
        print("[smoke] 动态发现工具：", names)
        assert names == ["add", "echo"], f"工具发现异常：{names}"

        by_name = {t.name: t for t in tools}

        result_add = await by_name["add"].ainvoke({"a": 3, "b": 4})
        print("[smoke] add(3, 4) =", result_add)
        assert "7" in str(result_add), f"add 返回异常：{result_add}"

        # 连续两次调用：若 server 被复用，其日志中只会出现一次启动记录
        result_echo_1 = await by_name["echo"].ainvoke({"text": "中文编码测试"})
        result_echo_2 = await by_name["echo"].ainvoke({"text": "第二次调用"})
        print("[smoke] echo =", result_echo_1)
        assert "中文编码测试" in str(result_echo_1), f"中文透传异常：{result_echo_1}"
        assert "第二次调用" in str(result_echo_2), f"会话复用异常：{result_echo_2}"

    print("[smoke] SMOKE PASS")


if __name__ == "__main__":
    asyncio.run(main())
