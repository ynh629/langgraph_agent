"""
guardrails/tool_guard.py
========================
工具入参校验：在「执行前」用 schema 校验工具调用参数，拦截非法/越界请求。

面试可讲：LLM 生成的工具参数是不可信输入。业务层（order_store）虽然也做了兜底校验，
但那是在执行时刻；这里在「调用前」用 Pydantic 声明式校验，把错误挡在副作用发生之前，
两道防线互补。
"""
from pydantic import BaseModel, Field, ValidationError


class RefundArgs(BaseModel):
    """apply_refund 的入参约束。"""

    order_id: str = Field(pattern=r"^[A-Za-z]{2}\d{6,}$", description="订单号，如 SO20260902002")
    amount: float = Field(gt=0, le=100000, description="退款金额（元），须大于 0")


# 工具名 → 入参 schema（未登记的工具默认放行）
_TOOL_SCHEMAS = {
    "apply_refund": RefundArgs,
}


def validate_tool_call(name: str, args: dict) -> tuple[bool, str]:
    """校验单个工具调用的入参。返回 (是否合法, 原因)。"""
    schema = _TOOL_SCHEMAS.get(name)
    if schema is None:
        return True, ""
    try:
        schema(**(args or {}))
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(p) for p in first.get("loc", ())) or "args"
        return False, f"{name} 入参非法（{field}）：{first.get('msg')}"
    return True, ""


def validate_tool_calls(tool_calls) -> tuple[bool, str]:
    """批量校验一批工具调用（LangChain tool_calls 结构）。"""
    for call in tool_calls or []:
        ok, reason = validate_tool_call(call.get("name"), call.get("args") or {})
        if not ok:
            return False, reason
    return True, ""
