"""
mcp_servers/order_server.py
===========================
V3 订单 MCP Server（stdio）：把「订单/售后」相关业务能力暴露为 MCP 工具。

暴露工具（来自 business.py）：
  - get_order_status   订单精确查询
  - get_logistics      物流轨迹查询
  - apply_refund       执行退款（写操作，纯执行，审批由上层图负责）

与检索服务拆成两个 Server 的原因（面试可讲）：
  - 服务边界 = 权限边界。订单/退款是敏感写操作，独立部署便于单独限权与审计；
  - 也演示了「一个 Agent 聚合多个 MCP Server」的能力。

约定：
  - stdout 是 MCP 协议流，禁止任何裸 print；日志一律走 stderr。

手动自测：
    python mcp_servers/order_server.py
"""
import logging
import os
import sys

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_AGENT_DIR, os.path.join(_AGENT_DIR, "检索")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 本项目 common.logging_config 默认写 stdout，会污染 MCP 协议流，这里强制走 stderr
os.environ.setdefault("LOG_CONSOLE_STREAM", "stderr")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("order_server")

from fastmcp import FastMCP

import business

mcp = FastMCP("order")

mcp.tool(business.get_order_status)
mcp.tool(business.get_logistics)
mcp.tool(business.apply_refund)


if __name__ == "__main__":
    logger.info("order-server 启动（stdio）")
    mcp.run(transport="stdio", show_banner=False)
