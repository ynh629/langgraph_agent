"""
eval/rag_eval.py
================
检索层评估：对标注好的「query → 期望主题」，计算 Hit@1/Hit@3/Hit@5 与 MRR。

依赖 Chroma（首次构建索引会写 chroma_db）。受限沙箱可能拦截写入，届时抛
RetrievalUnavailable，由 run_eval 标记该层为 skipped，而不是静默通过。
"""
import json
import os
import sys

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_AGENT_DIR, os.path.join(_AGENT_DIR, "检索")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import business

try:
    from .metrics import mean_or_zero
except ImportError:  # 允许直接运行本文件
    from metrics import mean_or_zero

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_dataset.jsonl")


class RetrievalUnavailable(RuntimeError):
    """检索不可用（如 Chroma 数据库不可写）。"""


def load_dataset(path: str = DATASET) -> list[dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _norm(title_path) -> str:
    if isinstance(title_path, (list, tuple)):
        return " ".join(str(x) for x in title_path)
    return str(title_path or "")


def _first_hit_rank(items: list, keywords: list) -> int | None:
    for rank, item in enumerate(items, start=1):
        haystack = _norm(item.get("title_path")) + " " + str(item.get("text", ""))
        if any(keyword in haystack for keyword in keywords):
            return rank
    return None


def run(dataset_path: str = DATASET, top_k: int = 5, use_rerank: bool = True) -> dict:
    cases = load_dataset(dataset_path)
    rows = []
    hits1, hits3, hits5, mrrs = [], [], [], []

    for case in cases:
        try:
            items = business.search_product_knowledge(case["query"], top_k=top_k)
        except Exception as exc:  # noqa: BLE001 - 统一转成检索不可用
            raise RetrievalUnavailable(str(exc)) from exc

        rank = _first_hit_rank(items, case["expect"])
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "rank": rank,
                "top_titles": [_norm(i.get("title_path")) for i in items],
            }
        )
        hits1.append(rank is not None and rank <= 1)
        hits3.append(rank is not None and rank <= 3)
        hits5.append(rank is not None and rank <= 5)
        mrrs.append(1.0 / rank if rank else 0.0)

    summary = {
        "total": len(cases),
        "hit@1": mean_or_zero(hits1),
        "hit@3": mean_or_zero(hits3),
        "hit@5": mean_or_zero(hits5),
        "mrr": mean_or_zero(mrrs),
        "use_rerank": use_rerank,
    }
    return {"layer": "rag", "status": "ok", "summary": summary, "cases": rows}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
