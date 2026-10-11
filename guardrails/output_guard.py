"""
guardrails/output_guard.py
==========================
输出护栏：对模型回复做 PII 脱敏。

面试可讲：输出侧脱敏是「最小必要披露」的落地——即使模型从上下文里读到了手机号/
身份证，对外回复也必须打码，避免把敏感信息写进日志或回显给用户。
"""
import re

# 顺序敏感：先长后短，避免 18 位身份证被 16-19 位银行卡规则截走
_PATTERNS = [
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("bank_card", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]


def _mask_value(value: str) -> str:
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:3]}{'*' * (len(value) - 6)}{value[-3:]}"


def mask_pii(text: str) -> tuple[str, list[dict]]:
    """对文本做 PII 脱敏，返回 (脱敏后文本, 命中列表)。

    命中列表元素形如 {"type": "phone", "value": "138****0000"}（value 已脱敏）。
    """
    result = text or ""
    hits: list[dict] = []
    for label, pattern in _PATTERNS:
        def _repl(match, label=label):
            raw = match.group(0)
            masked = _mask_value(raw)
            hits.append({"type": label, "value": masked})
            return masked

        result = pattern.sub(_repl, result)
    return result, hits
