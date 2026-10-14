"""
business.py
===========
电商客服的【业务能力层】：纯 Python 实现，不依赖 langchain / langgraph / mcp。

为什么单独一层：
  - 工具（tool）只是业务能力对外的「适配器」。LangChain 的 @tool 与 MCP 的
    mcp.tool 是两种绑定方式，但底层调用的是同一批业务函数；
  - 业务逻辑与框架解耦后，既能被 V1/V2 的进程内 Agent 调用，也能被 V3 的
    MCP Server 调用，还能脱离 LLM 环境直接单测。

上层适配器：
  - tools.py                          → LangChain @tool（V1 / V2 进程内）
  - mcp_servers/retrieval_server.py   → FastMCP（V3 检索服务）
  - mcp_servers/order_server.py       → FastMCP（V3 订单服务）
"""
import json
import os
from difflib import SequenceMatcher
from threading import Lock

import order_store
import retriever_factory

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PRODUCTS_JSON = os.path.join(_BASE_DIR, "data", "ecommerce_products.json")

# 索引构建是重操作且非线程安全，加锁避免并发重复构建
_build_lock = Lock()

# 商品库懒加载缓存
_products = None


def _load_products() -> list[dict]:
    """懒加载结构化商品库。"""
    global _products
    if _products is None:
        with open(_PRODUCTS_JSON, "r", encoding="utf-8") as f:
            _products = json.load(f).get("products", [])
    return _products


def search_product_knowledge(query: str, top_k: int = 5) -> list[dict]:
    """
    在商品/服务知识库中做【语义检索】。
    内部自动完成 BM25+向量双重检索、RRF 融合与相关性重排，返回最相关的知识片段（含来源标题与相关分数）。
    适用于：查询退换货政策、优惠券规则、售后流程、配送物流、支付发票等需要理解语义的问题。
    参数：
      query: 用户的自然语言问题（如"这款商品能七天无理由退货吗"）。
      top_k: 返回的片段条数，默认 5，最大 8。
    """
    with _build_lock:
        return retriever_factory.search_product_knowledge(query, top_k=min(int(top_k), 8))


def get_product_detail(product_name: str) -> dict:
    """
    精确查询指定商品的【结构化信息】（价格、库存、规格、保修、是否支持七天无理由等）。
    与 search_product_knowledge 不同，本工具做精确匹配而非语义搜索。
    适用于：用户询问某商品的价格、库存、规格、保修期、退换条件等确定性问题。
    参数：
      product_name: 商品名称（如"星耀 X1 智能手机"、"清风 K3 变频空调"）。
    """
    products = _load_products()
    name = (product_name or "").strip()
    best = None
    best_score = 0.0
    for p in products:
        if p["name"] == name:
            score = 1.0
        elif name in p["name"] or p["name"] in name:
            score = 0.9
        else:
            score = SequenceMatcher(None, name, p["name"]).ratio()
        if score > best_score:
            best_score = score
            best = p
    if best is not None and best_score >= 0.5:
        return {
            "id": best["id"],
            "name": best["name"],
            "brand": best["brand"],
            "category": best["category"],
            "price": best["price"],
            "stock": best["stock"],
            "specs": best["specs"],
            "warranty": best["warranty"],
            "support_7day_return": best["support_7day_return"],
            "return_note": best["return_note"],
        }
    return {"found": False, "message": f"未找到商品：{product_name}"}


def _safe_eval(expression: str) -> float:
    """安全求值：仅允许数字、四则运算、括号与小数。"""
    import ast
    import operator

    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _walk(node):
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_walk(node.left), _walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_walk(node.operand))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持表达式中的 {type(node).__name__}")

    return _walk(ast.parse(expression, mode="eval"))


def calculate(expression: str) -> float:
    """
    安全计算数学表达式（仅支持数字、加减乘除、括号和小数）。
    适用于：算折扣后价格、满减后实付、差价、运费等。
    参数：
      expression: 数学表达式字符串，如 "(3999 * 0.9) - 200" 或 "599 - 100"。
    """
    return round(float(_safe_eval(expression)), 2)


def get_order_status(order_id: str) -> dict:
    """
    按订单号【精确查询】订单状态（商品、金额、下单日期、当前状态、是否已有退款记录）。
    适用于：用户询问"我的订单到哪了""订单能不能退""这个订单多少钱"等需要先确认订单信息的问题。
    参数：
      order_id: 订单号，如 SO20260901001。
    """
    order = order_store.get_order(order_id)
    if order is None:
        return {"found": False, "message": f"未找到订单：{order_id}"}
    return {
        "found": True,
        "order_id": order["order_id"],
        "product_name": order["product_name"],
        "quantity": order["quantity"],
        "amount": order["amount"],
        "status": order["status"],
        "order_date": order["order_date"],
        "refund_amount": order.get("refund_amount"),
        "refund_status": order.get("refund_status"),
    }


def get_logistics(order_id: str) -> dict:
    """
    按订单号查询【物流轨迹】最新节点。
    适用于：用户询问包裹到哪里了、什么时候能到、为什么还没发货。
    参数：
      order_id: 订单号，如 SO20260901001。
    """
    order = order_store.get_order(order_id)
    if order is None:
        return {"found": False, "message": f"未找到订单：{order_id}"}
    return {
        "found": True,
        "order_id": order["order_id"],
        "carrier": order.get("carrier", ""),
        "tracking_no": order.get("tracking_no", ""),
        "latest_track": order.get("latest_track", ""),
        "tracked_at": order.get("tracked_at", ""),
    }


def apply_refund(order_id: str, amount: float) -> dict:
    """
    对指定订单【执行退款】（会真实写入订单数据）。金额需大于 0 且不超过订单实付金额，同一订单不可重复退款。
    本工具只负责执行；是否需要人工审批由上层流程决定（售后专员流程中，超过审批阈值的退款会先转人工）。
    参数：
      order_id: 订单号，如 SO20260901001。
      amount: 退款金额（元）。
    """
    return order_store.execute_refund(order_id, amount)
