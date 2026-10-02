"""
mcp_servers/echo_server.py
==========================
P0 兼容性冒烟用的最小 MCP Server（stdio transport）。

全项目 MCP Server 通用约定：
  - stdout 是 MCP 协议流，绝对禁止写入任何非协议内容（一句裸 print 就会让 client 解析失败）；
  - 所有日志一律走 stderr（logging 的 StreamHandler 显式绑定 sys.stderr）；
  - 需要输出请用 logger，不要用 print。

运行（由 client 以子进程方式拉起，也可手动自测）：
    python mcp_servers/echo_server.py
"""
import logging
import sys

from fastmcp import FastMCP

# 日志全部走 stderr：绝不污染 stdout 的协议流
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("echo_server")

mcp = FastMCP("echo-smoke")


@mcp.tool
def add(a: int, b: int) -> int:
    """返回两数之和（冒烟测试用）。"""
    logger.info("调用 add(a=%s, b=%s)", a, b)
    return a + b


@mcp.tool
def echo(text: str) -> str:
    """原样回显输入文本（用于验证中文与编码透传）。"""
    logger.info("调用 echo(text=%s)", text)
    return f"echo: {text}"


if __name__ == "__main__":
    logger.info("echo-server 启动（stdio）")
    mcp.run(transport="stdio")
