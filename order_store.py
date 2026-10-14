"""
order_store.py
==============
订单数据的读写与退款执行（纯 Python，不依赖 langchain / langgraph）。

为什么单独拆一个模块：
  1. 业务数据访问与「工具壳」解耦 —— tools.py 里的 @tool 只是薄薄一层包装，
     真正的订单逻辑在这里，可以脱离 LLM 环境直接单元测试；
  2. 退款是「写操作」，必须集中在一处并加锁，避免并发下重复退款。

数据文件：
  data/ecommerce_orders.seed.json  原始模板（只读，不修改）
  data/ecommerce_orders.json       运行时文件（退款会写回这里）

重置演示数据：
  python order_store.py --reset
"""
import argparse
import json
import os
from threading import Lock

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "data")
ORDERS_JSON = os.path.join(DATA_DIR, "ecommerce_orders.json")
SEED_JSON = os.path.join(DATA_DIR, "ecommerce_orders.seed.json")

_lock = Lock()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_orders(orders: list[dict]) -> None:
    with open(ORDERS_JSON, "w", encoding="utf-8") as f:
        json.dump({"orders": orders}, f, ensure_ascii=False, indent=2)


def list_orders() -> list[dict]:
    """读取全部订单。"""
    if not os.path.exists(ORDERS_JSON):
        reset_orders()
    return _read_json(ORDERS_JSON).get("orders", [])


def reset_orders() -> int:
    """用 seed 覆盖运行时文件，恢复初始状态。返回订单条数。"""
    with _lock:
        seed = _read_json(SEED_JSON).get("orders", [])
        _write_orders(seed)
    return len(seed)


def _normalize(order_id: str) -> str:
    return (order_id or "").strip().upper().replace(" ", "")


def get_order(order_id: str) -> dict | None:
    """按订单号精确查询，命中返回订单 dict，否则返回 None。"""
    oid = _normalize(order_id)
    if not oid:
        return None
    for o in list_orders():
        if _normalize(o.get("order_id", "")) == oid:
            return o
    return None


def execute_refund(order_id: str, amount: float) -> dict:
    """
    执行退款并写回运行时数据文件。校验：
      - 订单存在
      - 未重复退款
      - 退款金额 > 0 且不超过订单实付金额
    成功后将订单状态置为「退款中」并记录退款金额。
    """
    oid = _normalize(order_id)
    with _lock:
        orders = _read_json(ORDERS_JSON).get("orders", [])
        target = None
        for o in orders:
            if _normalize(o.get("order_id", "")) == oid:
                target = o
                break

        if target is None:
            return {"success": False, "message": f"未找到订单：{order_id}"}

        if target.get("refund_status"):
            return {
                "success": False,
                "message": f"订单 {target['order_id']} 已有退款记录（{target['refund_status']}），不可重复退款",
            }

        try:
            amt = round(float(amount), 2)
        except (TypeError, ValueError):
            return {"success": False, "message": f"退款金额格式不正确：{amount}"}

        if amt <= 0:
            return {"success": False, "message": "退款金额必须大于 0"}

        if amt > float(target["amount"]):
            return {
                "success": False,
                "message": f"退款金额 {amt} 元超过订单实付 {target['amount']} 元",
            }

        target["refund_amount"] = amt
        target["refund_status"] = "退款中"
        target["status"] = "退款中"
        _write_orders(orders)

    return {
        "success": True,
        "order_id": oid,
        "refund_amount": amt,
        "refund_status": "退款中",
        "message": f"退款 {amt} 元已受理，预计 1-3 个工作日原路退回",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="订单 mock 数据工具")
    parser.add_argument("--reset", action="store_true", help="用 seed 重置运行时订单数据")
    args = parser.parse_args()

    if args.reset:
        n = reset_orders()
        print(f"已重置订单数据，共 {n} 条")
    else:
        parser.print_help()
