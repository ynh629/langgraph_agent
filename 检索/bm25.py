import json
import os
import jieba
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple

import sys

# agent 目录与当前目录加入 sys.path，使 `common` 与本目录模块可被导入
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (BASE_DIR, HERE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.logging_config import get_logger
from common.exceptions import IndexError_

# 数据文件路径（位于 data/structured_segments.json）
DATA_JSON = os.path.join(BASE_DIR, "data", "structured_segments.json")

logger = get_logger(__name__)


class BM25Retriever:
    """
    基于 BM25 算法的关键词检索器。
    """
    def __init__(self, documents: List[str]):
        """
        初始化并构建 BM25 索引。
        """
        # 过滤空文档
        self.documents = [doc for doc in documents if doc and doc.strip()]
        if not self.documents:
            raise IndexError_(
                "文档列表为空，无法构建 BM25 索引",
                code="EMPTY_DOCUMENTS"
            )
        # 变量：分词后的文档列表（每个文档是一个词语列表）
        self.tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        # 变量：构建 BM25 模型（内部计算词频、文档频率等参数）
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info("BM25 索引构建完成，共 %d 篇文档", len(self.documents))

    def _tokenize(self, text: str) -> List[str]:
        # 使用 jieba 分词，默认精确模式
        try:
            tokens = jieba.lcut(text)
        except Exception as e:
            logger.error("jieba 分词失败: %s", e, exc_info=True)
            return []
        # 过滤掉空白词
        return [t.strip() for t in tokens if t.strip()]

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        根据查询词检索最相关的 top_k 个文档。
        返回：
            结果列表，每个元素是字典，包含：
                - doc_index: 原始文档列表中的索引
                - score: BM25 相关性分数（越高越相关）
                - text: 文档全文
        """
        if not query or not query.strip():
            logger.warning("BM25 收到空查询，返回空结果")
            return []

        # 变量：将查询文本分词
        query_tokens = self._tokenize(query)
        if not query_tokens:
            logger.warning("查询分词结果为空，query=%r", query[:50])
            return []

        # 变量：计算所有文档的 BM25 分数（numpy 数组）
        scores = self.bm25.get_scores(query_tokens)
        # 变量：按照分数降序排列，取 top_k 个索引
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            # 只返回分数大于0的结果（至少有一个关键词匹配）
            if scores[idx] > 0:
                results.append({
                    "doc_index": int(idx),
                    "score": float(scores[idx]),
                    "text": self.documents[idx]
                })

        # 若所有分数都为 0（无关键词匹配），返回分数最高的 top_k 个（保留部分结果）
        if not results and len(self.documents) > 0:
            logger.warning("BM25 无关键词匹配，返回分数最高的 %d 篇文档兜底", min(top_k, len(self.documents)))
            for idx in top_indices[:min(top_k, len(self.documents))]:
                results.append({
                    "doc_index": int(idx),
                    "score": float(scores[idx]),
                    "text": self.documents[idx]
                })

        logger.info("BM25 检索 query=%r 返回 %d 条结果", query[:50], len(results))
        return results


# --------------------------- 工具函数 ---------------------------
def load_documents_from_segments(json_path: str) -> List[str]:
    """
    从结构化片段 JSON 中提取所有文本块内容。
    """
    if not os.path.exists(json_path):
        raise IndexError_(
            f"数据文件不存在: {json_path}",
            code="DATA_FILE_NOT_FOUND"
        )
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    # 变量：过滤出 content 不为空的片段
    documents = [seg["content"] for seg in segments if seg.get("content") and seg["content"].strip()]
    logger.info("从 %s 加载了 %d 个文本块", json_path, len(documents))
    return documents


# --------------------------- 主程序 ---------------------------
if __name__ == "__main__":
    # 1. 加载文档（使用之前 parse_pdf.py 生成的 structured_segments.json）
    docs = load_documents_from_segments(DATA_JSON)

    # 2. 构建 BM25 索引
    bm25_retriever = BM25Retriever(docs)
    print("BM25 索引构建完成\n")

    # 3. 测试检索
    test_queries = [
        "净利润",
        "股东持股比例",
        "风险分析"
    ]
    for query in test_queries:
        print(f"查询：{query}")
        results = bm25_retriever.search(query, top_k=2)
        if not results:
            print("  无匹配结果")
        for i, res in enumerate(results):
            print(f"  Top {i+1} (BM25 分数={res['score']:.2f}):")
            # 截断文本前100字显示
            print(f"    内容：{res['text'][:100]}...")
        print("-" * 50)
