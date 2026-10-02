"""
policy.py
=========
业务策略（纯 Python，不依赖 langchain / langgraph，便于单元测试）

把「什么情况需要人工审批」这类规则从图编排代码里抽出来：
  - 阈值集中一处，调整风控口径不需要改图
  - 可脱离框架直接测试，验证成本低
"""

# 退款金额达到该阈值时必须转人工审批
REFUND_APPROVAL_THRESHOLD = 500.0


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def find_refund_calls(tool_calls) -> list:
    """从工具调用列表中筛出 apply_refund 调用。"""
    return [call for call in (tool_calls or []) if call.get("name") == "apply_refund"]


def needs_human_approval(tool_calls, threshold: float = REFUND_APPROVAL_THRESHOLD) -> bool:
    """这批工具调用里是否存在达到阈值、需要人工审批的退款申请。"""
    for call in find_refund_calls(tool_calls):
        if _to_float((call.get("args") or {}).get("amount")) >= threshold:
            return True
    return False


def extract_refund_request(tool_calls) -> tuple:
    """取第一笔退款申请，返回 (order_id, amount)；没有则返回 ("", 0.0)。"""
    calls = find_refund_calls(tool_calls)
    if not calls:
        return "", 0.0
    args = calls[0].get("args") or {}
    return str(args.get("order_id") or ""), _to_float(args.get("amount"))
