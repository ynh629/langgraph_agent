"""
verify_v2_logic.py
==================
V2 离线逻辑验证（不依赖 langgraph / langchain / LLM）。

覆盖：
  A. policy：审批阈值判定（含 500 元边界）与退款请求提取
  B. order_store：订单查询（含大小写/空格容错）
  C. order_store：退款写回、超金额拦截、重复退款拦截、非法金额拦截
  D. 结束后自动重置订单数据，保证可重复演示

运行：
    python scripts/verify_v2_logic.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import order_store  # noqa: E402
import policy  # noqa: E402


def _refund_call(order_id, amount):
    return {"name": "apply_refund", "args": {"order_id": order_id, "amount": amount}, "id": "call_1"}


def check_policy():
    print("=" * 64)
    print("A. policy：审批阈值判定")
    cases = [
        ([_refund_call("SO1", 59.9)], False, "小额退款不审批"),
        ([_refund_call("SO1", 499)], False, "499 元不审批"),
        ([_refund_call("SO1", 500)], True, "500 元触发审批（边界值）"),
        ([_refund_call("SO1", 3999)], True, "3999 元触发审批"),
        ([{"name": "get_order_status", "args": {"order_id": "SO1"}}], False, "非退款工具不审批"),
        ([], False, "无工具调用不审批"),
        ([_refund_call("SO1", "abc")], False, "金额非法按 0 处理，不审批"),
    ]
    for calls, expected, desc in cases:
        actual = policy.needs_human_approval(calls)
        assert actual == expected, f"[{desc}] 期望 {expected}，实际 {actual}"
        print(f"  ✓ {desc}：needs_human_approval={actual}")

    oid, amt = policy.extract_refund_request([_refund_call("SO20260901001", 3999)])
    assert oid == "SO20260901001" and amt == 3999.0, f"提取失败：{oid} {amt}"
    print(f"  ✓ 提取退款请求：order_id={oid}, amount={amt}")


def check_order_query():
    print("=" * 64)
    print("B. order_store：订单查询")
    order = order_store.get_order("SO20260902002")
    assert order is not None, "未找到订单 SO20260902002"
    assert order["amount"] == 59.9, f"金额不符：{order['amount']}"
    print(f"  ✓ 命中订单：{order['order_id']}｜{order['product_name']}｜{order['amount']} 元")

    loose = order_store.get_order("  so20260902002  ")
    assert loose is not None, "大小写/空格容错失败"
    print("  ✓ 大小写与空格容错正常")

    missing = order_store.get_order("SO00000000000")
    assert missing is None, "不存在的订单不应命中"
    print("  ✓ 不存在的订单返回 None")


def check_refund():
    print("=" * 64)
    print("C. order_store：退款执行与拦截")

    # C1 小额退款成功并写回
    result = order_store.execute_refund("SO20260902002", 59.9)
    assert result["success"] is True, f"退款应成功：{result}"
    print(f"  ✓ 小额退款成功：{result['message']}")

    reloaded = order_store.get_order("SO20260902002")
    assert reloaded["refund_status"] == "退款中", "退款状态未写回"
    assert reloaded["status"] == "退款中", "订单状态未写回"
    print(f"  ✓ 状态已写回文件：status={reloaded['status']}, refund_amount={reloaded['refund_amount']}")

    # C2 重复退款拦截
    again = order_store.execute_refund("SO20260902002", 10)
    assert again["success"] is False and "重复退款" in again["message"], f"重复退款未拦截：{again}"
    print(f"  ✓ 重复退款被拦截：{again['message']}")

    # C3 超过订单实付金额拦截
    over = order_store.execute_refund("SO20260901001", 5000)
    assert over["success"] is False and "超过" in over["message"], f"超额退款未拦截：{over}"
    print(f"  ✓ 超额退款被拦截：{over['message']}")

    # C4 非法金额拦截
    bad = order_store.execute_refund("SO20260901001", 0)
    assert bad["success"] is False, f"非法金额未拦截：{bad}"
    print(f"  ✓ 非法金额被拦截：{bad['message']}")

    # C5 seed 中已退款的订单不可再退
    dup = order_store.execute_refund("SO20260911011", 10)
    assert dup["success"] is False, f"已退款订单未拦截：{dup}"
    print(f"  ✓ 已退款订单被拦截：{dup['message']}")

    # C6 大额退款在数据层本身是允许的（审批是上层策略，不是数据层职责）
    large = order_store.execute_refund("SO20260901001", 3999)
    assert large["success"] is True, f"数据层大额退款应成功：{large}"
    print(f"  ✓ 大额退款数据层放行（审批由 policy 层负责）：{large['message']}")


def main():
    # 先重置运行态订单数据：本脚本对订单「初始状态」有断言，
    # 若不重置，验证结果就依赖上一次运行遗留的 data/ecommerce_orders.json。
    # CI 是全新检出，一旦该文件曾被演示/烟测写过（脏数据），门禁就会误报失败。
    # 确定性门禁必须自证初始态，不能依赖工作区文件。
    count = order_store.reset_orders()
    print("=" * 64)
    print(f"已重置订单数据（{count} 条），开始确定性验证")
    print("=" * 64)

    check_policy()
    check_order_query()
    check_refund()

    count = order_store.reset_orders()
    print("=" * 64)
    print(f"D. 已重置订单数据，共 {count} 条，演示可重复运行")
    print("=" * 64)
    print("V2 离线逻辑验证全部通过。")


if __name__ == "__main__":
    main()
