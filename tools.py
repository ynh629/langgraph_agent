"""
tools.py
========
把 business.py 的业务能力绑定为 LangChain 工具，供 V1 / V2 的【进程内】Agent 调用。

设计原则（对应面试讲解）：
  - 工具粒度 = 业务能力，不是内部模块。BM25 / 向量 / RRF / 重排 全部收敛在
    search_product_knowledge 内部，对外只暴露"按语义检索商品/服务知识片段"。
  - 本文件只做「绑定」，不含业务逻辑；同一批业务能力在 V3 里改由 MCP Server 暴露
    （见 mcp_servers/），这正是"框架解耦"的价值所在。

工具描述（LLM 看到的说明）来自 business.py 中各函数的 docstring。
"""
from langchain_core.tools import tool

import business

search_product_knowledge = tool(business.search_product_knowledge)
get_product_detail = tool(business.get_product_detail)
calculate = tool(business.calculate)
get_order_status = tool(business.get_order_status)
get_logistics = tool(business.get_logistics)
apply_refund = tool(business.apply_refund)


# 构成 Agent 的工具集。
# 售前专员：只有商品与价格相关能力，结构上拿不到订单与退款工具（工具权限隔离）
PRESALE_TOOLS = [search_product_knowledge, get_product_detail, calculate]

# 售后专员：可查订单、查物流、查售后政策、执行退款
AFTERSALE_TOOLS = [search_product_knowledge, get_order_status, get_logistics, apply_refund, calculate]

# V1 单 agent 使用的全量工具集（保持向后兼容）
AGENT_TOOLS = [search_product_knowledge, get_product_detail, get_order_status, get_logistics, apply_refund, calculate]
