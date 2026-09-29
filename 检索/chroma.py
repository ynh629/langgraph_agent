import json
import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional

import sys

# agent 目录与当前目录加入 sys.path，使 `common` 与本目录模块可被导入
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (BASE_DIR, HERE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.logging_config import get_logger
from common.exceptions import DocumentError, IndexError_
from model_paths import resolve_model

# agent 包根目录（检索 的上一级）
# 数据文件路径（位于 rag_system/data/structured_segments.json）
DATA_JSON = os.path.join(BASE_DIR, "data", "structured_segments.json")
# Chroma 数据库路径（位于 rag_system/chroma_db）
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

logger = get_logger(__name__)


# --------------------------- 1、引入数据 ---------------------------
def load_segments(json_path: str) -> List[Dict]:
    """从 JSON 加载结构化片段，返回片段列表。"""
    if not os.path.exists(json_path):
        raise DocumentError(
            f"数据文件不存在: {json_path}",
            code="DATA_FILE_NOT_FOUND"
        )
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
    except json.JSONDecodeError as e:
        raise DocumentError(
            f"数据文件 JSON 格式错误: {json_path}",
            code="DATA_FILE_INVALID_JSON",
            detail=str(e)
        )
    except OSError as e:
        raise DocumentError(
            f"读取数据文件失败: {json_path}",
            code="DATA_FILE_READ_ERROR",
            detail=str(e)
        )

    if not isinstance(segments, list):
        raise DocumentError(
            f"数据文件顶层应为列表: {json_path}",
            code="DATA_FILE_INVALID_FORMAT"
        )

    # 变量：筛选出 content 非空的片段
    valid_segments = [seg for seg in segments if seg.get("content") and seg["content"].strip()]
    logger.info("成功加载 %d 个有效片段（共 %d 条记录）", len(valid_segments), len(segments))
    return valid_segments


# --------------------------- 2. 构建向量索引 ---------------------------
def build_chroma_index(segments: List[Dict], collection_name: str = "ecommerce_knowledge"):
    """
    使用 Chroma 构建向量索引。
    返回：collection 对象，可用于检索。
    """
    if not segments:
        raise IndexError_(
            "片段列表为空，无法构建向量索引",
            code="EMPTY_SEGMENTS"
        )

    # 创建 Chroma 客户端（数据会持久化到 ../chroma_db 目录）
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
    except Exception as e:
        raise IndexError_(
            "无法创建 Chroma 客户端（数据库可能被其他进程锁定）",
            code="CHROMA_CLIENT_ERROR",
            detail=str(e)
        )

    # 定义 embedding 函数（BGE 中文小模型；resolve_model 优先取本地缓存，离线可用）
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=resolve_model("BAAI/bge-small-zh-v1.5")
    )

    # 如果 collection 已存在，先删除（可选，这里为了重建）
    try:
        client.delete_collection(collection_name)
        logger.info("已删除旧 collection: %s", collection_name)
    except Exception as e:
        logger.warning("删除旧 collection 失败（可能不存在）: %s", e)

    # 创建 collection（类似关系数据库的表）
    try:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"description": "电商商品与服务知识库向量库"}
        )
    except Exception as e:
        raise IndexError_(
            f"创建 collection 失败: {collection_name}",
            code="CHROMA_CREATE_ERROR",
            detail=str(e)
        )

    # 准备批量数据
    ids = []
    documents = []
    metadatas = []

    for i, seg in enumerate(segments):
        ids.append(f"chunk_{i}")
        documents.append(seg["content"])
        # 存储标题路径为元数据，方便检索后查看来源
        title_path = seg.get("title_path", [])
        metadatas.append({
            "title_path": " > ".join(title_path) if title_path else "无标题",
            "level": seg.get("level", 0)
        })

    # 批量添加文档
    try:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
    except Exception as e:
        raise IndexError_(
            "批量添加文档到 Chroma 失败（可能 embedding 计算出错）",
            code="CHROMA_ADD_ERROR",
            detail=str(e)
        )

    logger.info("索引构建完成，共 %d 个向量", len(documents))
    return collection


def get_or_create_collection(collection_name: str = "ecommerce_knowledge"):
    """
    获取已存在的 collection，若不存在则创建（用于增量更新场景）。
    返回：collection 对象。
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
    except Exception as e:
        raise IndexError_(
            "无法创建 Chroma 客户端",
            code="CHROMA_CLIENT_ERROR",
            detail=str(e)
        )

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=resolve_model("BAAI/bge-small-zh-v1.5")
    )

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"description": "电商商品与服务知识库向量库"}
        )
        logger.info("获取/创建 collection: %s", collection_name)
        return collection
    except Exception as e:
        raise IndexError_(
            f"获取/创建 collection 失败: {collection_name}",
            code="CHROMA_GET_OR_CREATE_ERROR",
            detail=str(e)
        )


# --------------------------- 3. 检索函数 ---------------------------
def search_similar(query: str, collection, top_k: int = 3):
    """
    在 collection 中搜索与 query 最相似的 top_k 个文本块。
    返回：查询结果列表，包含 id, content, metadata, distance
    """
    if not query or not query.strip():
        logger.warning("search_similar 收到空查询")
        return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results
    except Exception as e:
        logger.error("Chroma 查询失败: %s", e, exc_info=True)
        return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}


# --------------------------- 4. 主程序 ---------------------------
if __name__ == "__main__":
    # 步骤1：加载片段
    segments = load_segments(DATA_JSON)

    # 步骤2：构建索引
    collection = build_chroma_index(segments)

    # 步骤3：测试检索
    test_queries = [
        "公司2025年的净利润是多少？",
        "有哪些股东持有5%以上的股份？",
        "公司面临的主要风险有哪些？"
    ]

    print("\n" + "="*60)
    for query in test_queries:
        print(f"查询问题：{query}")
        results = search_similar(query, collection, top_k=2)
        for i, (doc_id, distance, doc_text, metadata) in enumerate(zip(
            results["ids"][0],
            results["distances"][0],
            results["documents"][0],
            results["metadatas"][0]
        )):
            print(f"  第{i+1}个结果（距离：{distance:.3f}）:")
            print(f"    标题路径：{metadata.get('title_path', '无')}")
            print(f"    内容预览：{doc_text[:100]}...")
        print("-"*60)
