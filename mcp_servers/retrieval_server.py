"""
mcp_servers/retrieval_server.py
===============================
V3 检索 MCP Server（stdio）：把「商品/服务知识检索」相关业务能力暴露为 MCP 工具。

暴露工具（来自 business.py）：
  - search_product_knowledge  语义检索（内部 BM25+向量+RRF+重排）
  - get_product_detail        商品结构化精确查询
  - calculate                 安全四则运算

约定：
  - stdout 是 MCP 协议流，禁止任何裸 print；日志一律走 stderr；
  - 会话常驻（client 侧用 client.session + load_mcp_tools），
    因此索引与模型的懒加载缓存可在多次工具调用间复用。

手动自测：
    python mcp_servers/retrieval_server.py
"""
import logging
import os
import sys

# 离线优先：本地已缓存 BGE 模型，跳过 HuggingFace 联网检查（必须在导入检索链之前）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 关键：本项目 common.logging_config 默认把日志写到 stdout，会污染 MCP 协议流，
# 这里强制改走 stderr（必须在导入 business / 检索链之前设置）。
os.environ.setdefault("LOG_CONSOLE_STREAM", "stderr")

# agent 根目录与 检索 目录加入 sys.path，使 business / 检索模块可被导入
_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_AGENT_DIR, os.path.join(_AGENT_DIR, "检索")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 日志全部走 stderr，绝不污染 stdout 的协议流
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("retrieval_server")

from fastmcp import FastMCP

import business

mcp = FastMCP("retrieval")

# 直接注册业务函数：名称与 docstring 自动成为 MCP 工具名与描述
mcp.tool(business.search_product_knowledge)
mcp.tool(business.get_product_detail)
mcp.tool(business.calculate)


if __name__ == "__main__":
    logger.info("retrieval-server 启动（stdio）")
    mcp.run(transport="stdio", show_banner=False)
