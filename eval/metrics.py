"""
eval/metrics.py
===============
纯函数指标：与框架无关，便于单测与复用。

检索指标（relevant / retrieved 均为 id 集合或列表）：
  - hit@k         ：top-k 内是否命中任一相关项
  - recall@k      ：top-k 内命中的相关项 / 全部相关项
  - precision@k   ：top-k 内命中的相关项 / k
  - mrr           ：第一个命中项排名的倒数
分类/判定指标：
  - accuracy / detection_rate / false_positive_rate
"""
from statistics import mean


def hit_at_k(retrieved: list, relevant, k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def recall_at_k(retrieved: list, relevant, k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list, relevant, k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / k


def mrr(retrieved: list, relevant) -> float:
    relevant = set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean_or_zero(values: list) -> float:
    return round(mean(values), 4) if values else 0.0


def accuracy(flags: list) -> float:
    return mean_or_zero([1.0 if f else 0.0 for f in flags])


def detection_rate(results: list) -> float:
    """恶意样本被正确拦截的比例（召回）。"""
    return mean_or_zero([1.0 if r else 0.0 for r in results])


def false_positive_rate(results: list) -> float:
    """正常样本被误拦的比例。"""
    return mean_or_zero([1.0 if r else 0.0 for r in results])
