"""
eval/judge.py
=============
LLM-as-judge（可选层）：用真实 LLM 对客服回答的「准确性/完整性/礼貌度」打分（1-5）。

注意：本层依赖 .env 中的真实 LLM，结果会受模型波动影响，默认不在 CI 门禁中启用，
仅作为人工抽检/离线报告使用（reasoning-first：先给理由再给分，降低武断评分）。
"""
import asyncio
import re
import sys
import os

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_AGENT_DIR, os.path.join(_AGENT_DIR, "检索")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_core.messages import HumanMessage, SystemMessage

JUDGE_PROMPT = """你是严格的电商客服质检员。请从「准确性（是否基于给定事实、无编造）、
完整性（是否回答了用户诉求）、礼貌度」三个维度评价客服回答。
先写一句理由，再给分。最后一行必须严格为：分数：<1-5 的整数>"""


async def judge(question: str, answer: str, context: str = "") -> dict:
    """对单条回答打分。返回 {"score": int|None, "reason": str, "raw": str}。"""
    from v3_agent import get_llm

    llm = get_llm()
    payload = f"用户问题：{question}\n参考事实：{context or '（无）'}\n客服回答：{answer}"
    response = await llm.ainvoke(
        [SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=payload)]
    )
    text = response.content if isinstance(response.content, str) else str(response.content)
    match = re.search(r"分数[：:]\s*([1-5])", text)
    return {"score": int(match.group(1)) if match else None, "reason": text.strip(), "raw": text}


async def run(samples: list[dict]) -> dict:
    """samples: [{"id","question","answer","context"}]。返回该层结果。"""
    rows = []
    scores = []
    for sample in samples:
        try:
            result = await judge(sample["question"], sample["answer"], sample.get("context", ""))
        except Exception as exc:  # noqa: BLE001 - LLM 不可用时该层整体跳过
            return {"layer": "judge", "status": "skipped", "reason": str(exc), "cases": rows}
        rows.append({"id": sample["id"], **result})
        if result["score"] is not None:
            scores.append(result["score"])

    average = round(sum(scores) / len(scores), 3) if scores else None
    return {
        "layer": "judge",
        "status": "ok",
        "summary": {"total": len(rows), "average_score": average},
        "cases": rows,
    }
