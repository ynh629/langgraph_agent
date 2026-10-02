"""
mcp_client.py
=============
V3 客户端：以 stdio 聚合「检索」「订单」两个 MCP Server，
把动态发现的工具按【角色名白名单】分组，供 V3 多智能体复用 V2 的权限隔离设计。

为什么分组要用白名单而不是全量：
  - MCP 是动态发现，server 端多加一个工具，agent 就会"顺手"拿到新权限；
  - 白名单把「工具发现」与「权限授予」解耦：发现是动态的，授权是显式的；
  - 分组结果与 V2 的 tools.py 中 PRESALE_TOOLS / AFTERSALE_TOOLS 完全对齐，
    这样 V3 与 V2 的专员能力边界一致，便于 verify_v3.py 做语义对照。

用法：
    async with MCPToolHub() as hub:
        presale_llm = llm.bind_tools(hub.presale_tools)
        aftersale_llm = llm.bind_tools(hub.aftersale_tools)

注意：MCP 工具与「会话」绑定，必须在上面的 async with 作用域内调用；
      会话关闭后工具即失效（这正是不能用一次性函数返回工具列表的原因）。
"""
import logging
import os
import sys
from contextlib import AsyncExitStack

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger(__name__)

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 角色 → 工具名白名单（与 tools.py 的 PRESALE_TOOLS / AFTERSALE_TOOLS 一一对应）
PRESALE_TOOL_NAMES = ("search_product_knowledge", "get_product_detail", "calculate")
AFTERSALE_TOOL_NAMES = (
    "search_product_knowledge",
    "get_order_status",
    "get_logistics",
    "apply_refund",
    "calculate",
)
# 全量白名单（去重，保持声明顺序）
ALL_TOOL_NAMES = tuple(dict.fromkeys(PRESALE_TOOL_NAMES + AFTERSALE_TOOL_NAMES))


def _server(script: str) -> dict:
    """单个 stdio MCP Server 的启动配置。"""
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
            # 关键：让 server 端日志走 stderr，避免污染 stdout 的 MCP 协议流
            "LOG_CONSOLE_STREAM": "stderr",
            # 离线优先：跳过 HuggingFace 联网检查
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    }


def build_config() -> dict:
    """两个 MCP Server 的聚合配置。"""
    return {
        "retrieval": _server("retrieval_server.py"),
        "order": _server("order_server.py"),
    }


class MCPToolHub:
    """聚合多个 MCP Server 的常驻会话，并按角色白名单分组工具。

    实现要点：用 AsyncExitStack 同时持有多个 server 会话，
    会话在 __aenter__ 打开、__aexit__ 统一关闭，中间暴露分好组的工具。
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or build_config()
        self._stack = AsyncExitStack()
        self._tools: dict = {}

    async def __aenter__(self) -> "MCPToolHub":
        client = MultiServerMCPClient(self._config)
        for server_name in self._config:
            session = await self._stack.enter_async_context(client.session(server_name))
            for t in await load_mcp_tools(session):
                self._tools[t.name] = t
        self._validate()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._stack.aclose()

    def _validate(self) -> None:
        discovered = set(self._tools)
        missing = set(ALL_TOOL_NAMES) - discovered
        if missing:
            raise RuntimeError(f"MCP 工具缺失（server 未按约定暴露）：{sorted(missing)}")
        extra = discovered - set(ALL_TOOL_NAMES)
        if extra:
            # 白名单外的工具不授予任何角色，仅告警，不阻断主流程
            logger.warning("发现白名单外的 MCP 工具，已忽略：%s", sorted(extra))

    def _group(self, names) -> list:
        return [self._tools[n] for n in names]

    @property
    def all_tools(self) -> list:
        return self._group(ALL_TOOL_NAMES)

    @property
    def presale_tools(self) -> list:
        """售前专员工具：商品检索 / 商品详情 / 计算（无订单与退款能力）。"""
        return self._group(PRESALE_TOOL_NAMES)

    @property
    def aftersale_tools(self) -> list:
        """售后专员工具：售后政策检索 / 订单 / 物流 / 退款 / 计算。"""
        return self._group(AFTERSALE_TOOL_NAMES)
