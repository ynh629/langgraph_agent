"""
eval/guardrail_eval.py
======================
护栏层评估：
  - 攻击样本拦截率（召回）：恶意输入应被拦下；
  - 误拦率（误报）：正常输入不应被拦；
  - PII 脱敏演示：展示输出侧脱敏效果。
"""
import json
import os
import sys

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from guardrails import detect_injection, mask_pii

try:
    from .metrics import detection_rate, false_positive_rate
except ImportError:  # 允许直接运行本文件
    from metrics import detection_rate, false_positive_rate

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adversarial.jsonl")

_PII_SAMPLE = "请联系客服 13812345678 或 admin@shop.com，客户身份证号 11010119900307123X。"


def load_dataset(path: str = DATASET) -> list[dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run(dataset_path: str = DATASET) -> dict:
    cases = load_dataset(dataset_path)
    malicious, benign, rows = [], [], []

    for case in cases:
        result = detect_injection(case["text"])
        blocked = not result.allowed
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "expected_blocked": case["blocked"],
                "blocked": blocked,
                "matched": result.category or result.matched or "",
            }
        )
        (malicious if case["blocked"] else benign).append(blocked)

    masked, hits = mask_pii(_PII_SAMPLE)
    summary = {
        "total": len(cases),
        "attack_detection_rate": detection_rate(malicious),
        "false_positive_rate": false_positive_rate(benign),
        "malicious_total": len(malicious),
        "benign_total": len(benign),
    }
    return {
        "layer": "guardrail",
        "status": "ok",
        "summary": summary,
        "cases": rows,
        "pii_demo": {"input": _PII_SAMPLE, "masked": masked, "hits": hits},
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
