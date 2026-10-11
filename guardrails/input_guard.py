"""
guardrails/input_guard.py
=========================
输入护栏：拦截 prompt injection / jailbreak / 提示词窃取 / 越权审批诱导。

设计取舍（面试可讲）：
  - 这里用「规则匹配」而非分类模型：规则零依赖、零延迟、可解释，适合作为第一道
    廉价闸门；真正的对抗场景应叠加分类器或 LLM 复核（见 docs/security.md 的演进建议）。
  - 规则命中即拦截并给出 category，便于在评估集里统计各类攻击的召回率。
"""
import re
from dataclasses import dataclass


@dataclass
class GuardResult:
    """护栏判定结果。allowed=False 时携带命中的规则类别与原文片段。"""

    allowed: bool
    category: str = ""
    matched: str = ""
    reason: str = ""


# (正则, 类别)。类别用于评估统计与日志归因。
_INJECTION_PATTERNS = [
    # 经典 prompt injection
    (r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", "prompt_injection"),
    (r"disregard\s+(the\s+)?(system|previous|above)", "prompt_injection"),
    (r"忽略.{0,8}(指令|规则|提示|设定|要求)", "prompt_injection"),
    (r"(无视|绕过|跳过)(以上|之前|系统|所有)?(的)?(指令|规则|限制|安全)", "prompt_injection"),
    # jailbreak / 角色劫持
    (r"(你现在是|从现在起你是|从现在开始你)(一个)?(不受限制|没有限制|无限制|越狱)", "jailbreak"),
    (r"(developer|debug|god|dan)\s*mode", "jailbreak"),
    (r"do\s+anything\s+now", "jailbreak"),
    (r"扮演(一个)?(不受任何限制|没有道德|越狱)的", "jailbreak"),
    # 提示词窃取
    (r"(输出|告诉我|泄露|打印|复述|展示)(你的)?(系统)?(提示词|prompt|设定|system\s*prompt)", "prompt_leak"),
    (r"(reveal|print|repeat|show|leak)\s+(your\s+)?(system\s+)?(prompt|instructions)", "prompt_leak"),
    # 越权审批诱导：试图绕过人工审批直接退款
    (r"(跳过|绕过|无需|不用|免)(人工)?审批", "approval_bypass"),
    (r"直接(批准|放行|通过|执行)退款.*(不要|无需|不用).*(审批|人工)", "approval_bypass"),
    (r"(批准|放行|通过).{0,6}退款.{0,6}(不用|无需|不要).{0,6}(审批|人工)", "approval_bypass"),
]


def detect_injection(text: str, extra_patterns=None) -> GuardResult:
    """检测输入是否包含注入/越狱/窃取/越权诱导。命中返回 allowed=False。"""
    content = text or ""
    for pattern, category in _INJECTION_PATTERNS + list(extra_patterns or []):
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return GuardResult(
                allowed=False,
                category=category,
                matched=match.group(0),
                reason=f"输入命中 {category} 规则，已拦截",
            )
    return GuardResult(allowed=True)
