"""
retriever_factory.py
====================
检索器工厂：懒加载 + 单例缓存，供 Agent 工具调用。

职责：
  1. 首次调用时从 data/ecommerce_segments.json 构建 BM25 索引 与 Chroma 向量索引；
  2. 组装 HybridRetriever（RRF 融合），可选挂载 BGE 交叉编码重排器；
  3. 进程内缓存实例，避免 Agent 多轮调用时重复建索引。

正确性说明：
  - embedding/重排模型走 model_paths.resolve_model()（本地缓存优先）；
  - 首次构建需下载/加载 BGE 中文模型，耗时较长，因此做成懒加载。
"""
import sys
import os

# 确保 agent 与 检索 目录可被导入
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_RETRIEVAL_DIR = os.path.join(_AGENT_DIR, "检索")
for _p in (_AGENT_DIR, _RETRIEVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chroma import load_segments, build_chroma_index
from bm25 import BM25Retriever
from hybrid_retriever import HybridRetriever
from rerank import Reranker

from common.logging_config import get_logger

# 电商知识库片段路径（结构化片段，含 content/title_path/level/page）
DATA_JSON = os.path.join(_AGENT_DIR, "data", "ecommerce_segments.json")

logger = get_logger(__name__)

_hybrid = None
_reranker = None
_candidates_k = 20       # 混检初筛候选数
_final_top_k = 5         # 返回给 LLM 的最终条数


def get_hybrid_retriever(use_rerank: bool = True):
    """懒加载并缓存 HybridRetriever（可选重排）。返回 HybridRetriever 实例。"""
    global _hybrid, _reranker
    if _hybrid is None:
        segments = load_segments(DATA_JSON)
        logger.info("开始构建索引，共 %d 个片段", len(segments))

        documents = [seg["content"] for seg in segments]
        bm25_retriever = BM25Retriever(documents)
        chroma_coll = build_chroma_index(segments)
        _hybrid = HybridRetriever(chroma_coll, bm25_retriever)

        if use_rerank:
            logger.info("正在加载 BGE Reranker 模型...")
            _reranker = Reranker(backend="bge")
        logger.info("检索器构建完成")
    return _hybrid


def search_product_knowledge(query: str, top_k: int = 5, use_rerank: bool = True) -> list:
    """
    对商品/服务知识库执行混合检索（BM25 + 向量 + RRF），可选重排。
    返回按相关性排序的片段列表。
    """
    hybrid = get_hybrid_retriever(use_rerank=use_rerank)
    candidates = hybrid.search(query, top_k=_candidates_k, method="rrf")
    logger.info("混合检索候选 %d 条", len(candidates))

    if not candidates:
        return []

    if use_rerank and _reranker is not None:
        results = _reranker.rerank(query, candidates, top_k=min(top_k, _final_top_k))
    else:
        results = candidates[: min(top_k, _final_top_k)]

    return [
        {
            "text": r["text"],
            "title_path": r.get("metadata", {}).get("title_path", "无标题"),
            "score": round(r.get("rerank_score") or r.get("score", 0.0), 4),
        }
        for r in results
    ]


if __name__ == "__main__":
    for q in ["这款商品的退换货政策是什么？", "优惠券如何使用？"]:
        print("查询：", q)
        for r in search_product_knowledge(q, top_k=3):
            print(f"  [{r['title_path']}] {r['text'][:80]}  score={r['score']}")