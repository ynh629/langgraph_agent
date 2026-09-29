# hybrid_retriever.py
import os
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple

import sys

# agent 目录与当前目录加入 sys.path，使 `common` 与本目录模块可被导入
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (BASE_DIR, HERE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chroma import load_segments, build_chroma_index, search_similar
from bm25 import BM25Retriever
from common.logging_config import get_logger
from common.exceptions import RetrievalError

# 数据文件路径（位于 data/structured_segments.json）
DATA_JSON = os.path.join(BASE_DIR, "data", "structured_segments.json")

logger = get_logger(__name__)


def _make_key(text: str, doc_id: Optional[str] = None) -> str:
    """
    生成用于融合去重的稳定键。
    优先使用文档 id；若无 id，则使用文本哈希 + 前 50 字符，降低碰撞风险。
    """
    if doc_id:
        return f"id:{doc_id}"
    return f"hash:{hashlib.md5(text.encode()).hexdigest()}:{text[:50]}"


class HybridRetriever:
    def __init__(self, chroma_collection, bm25_retriever: BM25Retriever):
        self.chroma_collection = chroma_collection
        self.bm25 = bm25_retriever

    def search(
        self,
        query: str,
        top_k: int = 5,
        method: str = "weighted",
        weights: Tuple[float, float] = (0.6, 0.4),
        candidate_k: int = 20  # 每种检索器初筛的候选数量
    ) -> List[Dict]:
        # 输入校验
        if not query or not query.strip():
            logger.warning("收到空查询，返回空结果")
            return []

        # 1. 向量检索，获取候选集
        vec_results = self._retrieve_from_chroma(query, candidate_k)
        # 2. BM25 检索，获取候选集
        bm25_results = self.bm25.search(query, top_k=candidate_k)  # 返回包含 doc_index, score, text

        logger.info(
            "混合检索 query=%r 向量召回=%d BM25召回=%d",
            query[:50], len(vec_results), len(bm25_results)
        )

        # 两个检索器都无结果时，直接返回空
        if not vec_results and not bm25_results:
            logger.warning("向量与BM25均无召回结果，query=%r", query[:50])
            return []

        # 4. 根据method融合
        if method == "weighted":
            merged = self._weighted_fusion(vec_results, bm25_results, weights)
        elif method == "rrf":
            merged = self._rrf_fusion(vec_results, bm25_results, k=60)
        else:
            raise RetrievalError(
                f"不支持的融合方法: {method}",
                code="UNSUPPORTED_METHOD"
            )

        # 5. 按融合分数降序排序，截取 top_k
        merged.sort(key=lambda x: x["score"], reverse=True)
        result = merged[:top_k]
        logger.info("混合检索完成，返回 %d 条结果", len(result))
        return result

    def _retrieve_from_chroma(self, query: str, n_results: int) -> List[Dict]:
        """调用 Chroma 检索，返回标准化结果列表。"""
        try:
            raw = self.chroma_collection.query(
                query_texts=[query],
                n_results=n_results
            )
        except Exception as e:
            logger.error("Chroma 查询失败: %s", e, exc_info=True)
            return []

        results = []
        try:
            ids = raw.get("ids", [[]])[0]
            distances = raw.get("distances", [[]])[0]
            documents = raw.get("documents", [[]])[0]
            metadatas = raw.get("metadatas", [None])[0]
        except (IndexError, KeyError, TypeError) as e:
            logger.error("Chroma 返回结果格式异常: %s", e, exc_info=True)
            return []

        if not ids:
            return []

        for i in range(len(ids)):
            # Chroma 返回的距离是余弦距离 (0~2)，越小越相似
            dist = distances[i] if i < len(distances) else 1.0
            # 转换为相似度（1 - 距离/2）映射到 [0,1]
            sim = 1.0 - dist / 2.0
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            results.append({
                "id": ids[i],
                "score": sim,          # 相似度，越大越好
                "text": documents[i] if i < len(documents) else "",
                "metadata": meta if isinstance(meta, dict) else {}
            })
        return results

    def _weighted_fusion(
        self,
        vec_results: List[Dict],
        bm25_results: List[Dict],
        weights: Tuple[float, float]
    ) -> List[Dict]:
        """加权分数融合。"""
        # 归一化向量相似度到 [0,1]
        vec_scores = np.array([r["score"] for r in vec_results]) if vec_results else np.array([])
        bm25_scores = np.array([r["score"] for r in bm25_results]) if bm25_results else np.array([])

        # Min-Max 归一化，空数组或全相同值时安全处理
        def safe_minmax(arr):
            if len(arr) == 0:
                return arr
            min_v, max_v = arr.min(), arr.max()
            if max_v == min_v:
                return np.ones_like(arr) * 0.5
            return (arr - min_v) / (max_v - min_v)

        vec_norm = safe_minmax(vec_scores)
        bm25_norm = safe_minmax(bm25_scores)

        # 构建以稳定键为标识的分数映射
        merged_dict = {}

        def add_to_dict(results, norm_scores, weight, source):
            for res, norm_score in zip(results, norm_scores):
                key = _make_key(res["text"], res.get("id"))
                if key not in merged_dict:
                    merged_dict[key] = {
                        "text": res["text"],
                        "metadata": res.get("metadata", {}),
                        "score": 0.0,
                        "source_scores": {}
                    }
                merged_dict[key]["score"] += weight * norm_score
                merged_dict[key]["source_scores"][source] = res["score"]

        add_to_dict(vec_results, vec_norm, weights[0], "vector")
        add_to_dict(bm25_results, bm25_norm, weights[1], "bm25")

        # 转换为列表
        merged = [
            {
                "id": key,
                "score": val["score"],
                "text": val["text"],
                "metadata": val["metadata"]
            }
            for key, val in merged_dict.items()
        ]
        return merged

    def _rrf_fusion(
        self,
        vec_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """倒数排名融合。"""
        # 按原始分数排序（降序）以获得排名
        vec_sorted = sorted(vec_results, key=lambda x: x["score"], reverse=True)
        bm25_sorted = sorted(bm25_results, key=lambda x: x["score"], reverse=True)

        rrf_scores = {}
        # 处理向量结果
        for rank, res in enumerate(vec_sorted, start=1):
            key = _make_key(res["text"], res.get("id"))
            rrf_scores.setdefault(key, {"text": res["text"], "metadata": res.get("metadata", {}), "score": 0.0})
            rrf_scores[key]["score"] += 1.0 / (k + rank)
        # 处理 BM25 结果
        for rank, res in enumerate(bm25_sorted, start=1):
            key = _make_key(res["text"], res.get("id"))
            rrf_scores.setdefault(key, {"text": res["text"], "metadata": res.get("metadata", {}), "score": 0.0})
            rrf_scores[key]["score"] += 1.0 / (k + rank)

        merged = [
            {"id": key, "score": val["score"], "text": val["text"], "metadata": val["metadata"]}
            for key, val in rrf_scores.items()
        ]
        return merged


# --------------------------- 主程序 ---------------------------
if __name__ == "__main__":
    # 1. 加载数据（复用 chroma.py 里的 load_segments）
    segments = load_segments(DATA_JSON)

    # 2. 构建 Chroma 向量索引
    print("正在构建向量索引...")
    chroma_coll = build_chroma_index(segments)

    # 3. 构建 BM25 索引
    print("正在构建 BM25 索引...")
    documents = [seg["content"] for seg in segments]
    bm25_retriever = BM25Retriever(documents)

    # 4. 创建混合检索器
    hybrid = HybridRetriever(chroma_coll, bm25_retriever)

    # 5. 测试查询
    test_queries = [
        "公司2025年的净利润是多少？",
        "有哪些股东持有5%以上的股份？",
        "公司面临的主要风险有哪些？"
    ]

    for method in ["weighted", "rrf"]:
        print(f"\n{'='*60}")
        print(f"融合策略：{method.upper()}")
        for query in test_queries:
            print(f"\n查询：{query}")
            results = hybrid.search(query, top_k=3, method=method)
            if not results:
                print("  无匹配结果")
            for i, res in enumerate(results):
                print(f"  Top {i+1} (融合分数={res['score']:.4f})")
                print(f"    标题路径：{res['metadata'].get('title_path', '无')}")
                print(f"    内容预览：{res['text'][:120]}...")
        print("-" * 60)
