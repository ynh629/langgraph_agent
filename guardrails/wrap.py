"""
guardrails/wrap.py
==================
把「工具入参校验」（tool_guard）挂到工具执行前的最后一刻：用结构化工具重新包一层，
先校验参数、再委托原工具执行。

为什么包在工具上而不是改图（面试可讲）：
  入参校验是横切关注点。包在工具壳上即可对任意图（V2/V3/V4）生效，无需侵入图结构，
  也保证了 V2/V3 源码不变；被拦截的调用不会产生任何副作用（如误退款）。
"""
from langchain_core.tools import StructuredTool

from .tool_guard import validate_tool_call


def guard_tool(tool):
    """返回一个「先校验入参再执行」的工具；无法包装时原样返回，绝不阻断主流程。

    注意：不要继承原工具的 response_format。MCP 适配器工具会自行处理
    content_and_artifact，若把该字段搬到标准 StructuredTool 上会触发二段校验，
    而其返回的内容列表不满足「(content, artifact) 二元组」约定而报错。
    包装层统一用默认的 content 格式，直接把原结果透传。
    """
    name = getattr(tool, "name", None)
    if not name:
        return tool

    async def _arun(**kwargs):
        ok, reason = validate_tool_call(name, kwargs)
        if not ok:
            return f"[护栏拦截] {reason}"
        return await tool.ainvoke(kwargs)

    try:
        return StructuredTool.from_function(
            coroutine=_arun,
            name=name,
            description=getattr(tool, "description", "") or name,
            args_schema=getattr(tool, "args_schema", None),
        )
    except Exception:  # noqa: BLE001 - 包装失败不应影响可用性
        return tool


def guard_tools(tools) -> list:
    """批量包装。"""
    return [guard_tool(t) for t in tools]
