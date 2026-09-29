# 文本切分策略模块
import json
import os
from typing import Dict, List, Optional

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import sys

# agent 目录与上一级加入 sys.path，使 `common` 可被导入
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (BASE_DIR, HERE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.logging_config import get_logger
from common.exceptions import DocumentError

logger = get_logger(__name__)

# 数据文件路径（电商知识库片段）
DATA_JSON = os.path.join(BASE_DIR, "data", "ecommerce_segments.json")

def load_text_from_segments(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
        # 变量：将所有片段的 content 用双换行连接，保留段落边界
    full_text = "\n\n".join(seg["content"] for seg in segments if seg["content"].strip())
    return full_text
#---------------------RecursiveCharacterTextSplitter递归按分隔符切分------------------
def run_recursive_splitter(text,chunk_size=1024,chunk_overlap=200):  #递归字符拆分器
    separators = ["\n\n", "\n", "。", ".", "；",]  #优先分隔列表
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,  #块大小
        chunk_overlap=chunk_overlap,  #块重叠
        length_function=len
    )
    docs = splitter.create_documents([text])
    chunks=[doc.page_content for doc in docs]
    print(f"RecursiveSplitter (size={chunk_size}, overlap={chunk_overlap})")
    print(f"  块数量: {len(chunks)}")
    if chunks:
        print(f"  平均长度: {np.mean([len(c) for c in chunks]):.0f} 字符")
        print(f"  示例块前100字: {chunks[0][:100]}...")
    else:
        print("  未产生任何块")
    return chunks
#--------------------------语义切片，根据向量embedding差异---------------------
def semantic_chunking(text,model_name="BAAI/bge-small-zh-v1.5",threshold=0.5):
    if not text or not text.strip():
        return []
    model=SentenceTransformer(model_name)
    sentences = text.replace("?", "。").replace("!", "。").split("。")
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <=1:
        return [text]
    embeddings=model.encode(sentences)
    similarities=[]
    for i in range(len(embeddings)-1):
        sim=cosine_similarity([embeddings[i]],[embeddings[i+1]])[0][0]
        similarities.append(sim)
    breakpoints=[0]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            breakpoints.append(i+1)
    breakpoints.append(len(sentences))
    chunks=[]
    for start,end in zip(breakpoints[:-1], breakpoints[1:]):
        chunk_text="。".join(sentences[start:end])
        chunks.append(chunk_text)
    print(f"SemanticChunker (threshold={threshold})")
    print(f"  块数量: {len(chunks)}")
    if chunks:
        print(f"  平均长度: {np.mean([len(c) for c in chunks]):.0f} 字符")
        print(f"  示例块前100字: {chunks[0][:100]}...")
    else:
        print("  未产生任何块")
    return chunks
# ----------------------- 供 QA 系统调用的高级切分入口 -------------------------
def apply_advanced_splitting(
    segments_json_path: str,
    output_json_path: str,
    method: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    threshold: float = 0.5,
    model_name: str = "BAAI/bge-small-zh-v1.5",
) -> List[Dict]:
    """
    在 parse_pdf 之后调用：读取结构化片段 JSON，按用户选择的切分策略重新切分，
    并将结果保存为新的 JSON（格式与 structured_segments.json 兼容，供 QA 系统加载）。

    参数：
        segments_json_path: 输入 JSON 路径（parse_pdf.py 生成的 structured_segments.json）
        output_json_path: 输出 JSON 路径（切分后的新片段，如 chunks_recursive.json）
        method: 切分策略，"recursive"（递归字符切分）或 "semantic"（语义切分）
        chunk_size: recursive 模式的块大小（默认 1000）
        chunk_overlap: recursive 模式的块重叠（默认 100）
        threshold: semantic 模式的相似度阈值（默认 0.5，越小块越大）
        model_name: semantic 模式使用的 embedding 模型

    返回：
        切分后的片段列表。每个片段包含：
            - title_path: 尽量映射回原始片段的标题路径（基于字符偏移量匹配）
            - content: 切分后的文本块
            - level: 原始标题级别（跨边界时置 0）
            - page: 尽量继承原始片段页码
            - chunk_method / chunk_params: 本次切分的方法与参数
    """
    # 1. 加载并校验输入片段
    if not os.path.exists(segments_json_path):
        raise DocumentError(
            f"输入数据文件不存在: {segments_json_path}",
            code="DATA_FILE_NOT_FOUND"
        )
    try:
        with open(segments_json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
    except json.JSONDecodeError as e:
        raise DocumentError(
            f"输入数据文件 JSON 格式错误: {segments_json_path}",
            code="DATA_FILE_INVALID_JSON",
            detail=str(e)
        )
    valid_segments = [seg for seg in segments if seg.get("content") and seg["content"].strip()]
    if not valid_segments:
        raise DocumentError(
            f"输入 JSON 中没有有效片段: {segments_json_path}",
            code="EMPTY_SEGMENTS"
        )

    # 2. 拼接全文，并记录每个原始片段在全文中的字符边界 [start, end)
    parts = []
    boundaries = []  # 元素: (segment, start, end)
    pos = 0
    for seg in valid_segments:
        content = seg["content"].strip()
        parts.append(content)
        boundaries.append((seg, pos, pos + len(content)))
        pos += len(content)
        parts.append("\n\n")
        pos += 2
    full_text = "".join(parts)

    # 3. 按用户选择的方法切分
    if method == "recursive":
        chunks = run_recursive_splitter(
            full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunk_params = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    elif method == "semantic":
        chunks = semantic_chunking(full_text, model_name=model_name, threshold=threshold)
        chunk_params = {"threshold": threshold, "model_name": model_name}
    else:
        raise ValueError(f"未知的切分方法: {method}，可选 'recursive' 或 'semantic'")

    if not chunks:
        raise DocumentError(
            "切分结果为空，请检查输入文本",
            code="SPLIT_EMPTY_RESULT"
        )

    # 4. 将每个块映射回原始片段，继承 title_path / level / page
    def _find_chunk_position(chunk, start_pos):
        """尽力在 full_text 中定位 chunk 的起始位置（逐级降级匹配）。"""
        # 策略1：精确子串匹配（recursive 切分的块与原文完全一致）
        idx = full_text.find(chunk, start_pos)
        if idx != -1:
            return idx
        # 策略2：空白折叠后匹配（semantic 切分去掉了句子首尾空白）
        collapsed = "".join(chunk.split())
        tail = full_text[start_pos:]
        idx = tail.find(collapsed)
        if idx != -1:
            return start_pos + idx
        # 策略3：取前 50 字符做前缀匹配（处理 ?/! 被替换为 。导致的差异）
        prefix = collapsed[:50]
        if prefix:
            idx = tail.find(prefix)
            if idx != -1:
                return start_pos + idx
        return start_pos

    def _find_owner(chunk_start, chunk_end):
        """根据字符偏移找到 chunk 起始位置所属的原始片段，并判断是否跨边界。"""
        owner = None
        for seg, s, e in boundaries:
            if s <= chunk_start < e:
                owner = seg
                break
        if owner is None and boundaries:
            # chunk 落在分隔符上：取前一个结束位置最近的片段兜底
            prev = None
            for seg, s, e in boundaries:
                if e <= chunk_start:
                    prev = seg
                else:
                    break
            owner = prev if prev is not None else boundaries[0][0]
        crosses = sum(1 for _, s, e in boundaries if s < chunk_end and chunk_start < e) > 1
        return owner, crosses

    out_segments = []
    search_pos = 0
    for i, chunk in enumerate(chunks):
        chunk_start = _find_chunk_position(chunk, search_pos)
        chunk_end = chunk_start + len(chunk)
        search_pos = max(search_pos, chunk_start + 1)

        owner, crosses = _find_owner(chunk_start, chunk_end)
        title_path = list(owner.get("title_path", [])) if owner else [f"{method}_chunk_{i}"]
        out_segments.append({
            "title_path": title_path,
            "content": chunk,
            "level": 0 if crosses else owner.get("level", 0) if owner else 0,
            "page": owner.get("page") if owner else None,
            "chunk_method": method,
            "chunk_params": chunk_params,
        })

    # 5. 保存结果
    out_dir = os.path.dirname(os.path.abspath(output_json_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out_segments, f, ensure_ascii=False, indent=2)

    logger.info(
        "高级切分完成（method=%s，%d 个片段），结果保存到 %s",
        method, len(out_segments), output_json_path
    )
    print(f"\n切分完成（method={method}），共 {len(out_segments)} 个片段")
    print(f"结果已保存到: {output_json_path}")
    return out_segments


# -----------------------对比实验主程序--------------------------
if __name__ == "__main__":
    full_text = load_text_from_segments(DATA_JSON)

    print(f"原始文本总长度: {len(full_text)} 字符\n")
    # 测试不同参数的 RecursiveCharacterTextSplitter
    print("=" * 50)
    chunks_r_500 = run_recursive_splitter(full_text, chunk_size=500, chunk_overlap=50)
    print()
    chunks_r_1000 = run_recursive_splitter(full_text, chunk_size=1000, chunk_overlap=100)
    print()
    chunks_r_1500 = run_recursive_splitter(full_text, chunk_size=1500, chunk_overlap=150)
    print()
    # 测试语义切片
    print("=" * 50)
    chunks_sem = semantic_chunking(full_text, threshold=0.5)
    print()
    # 可尝试不同阈值，如 0.3、0.7，chunks_sem_high = semantic_chunking(full_text, threshold=0.7)

    # 简单对比
    print("=" * 50)
    print("对比摘要：")
    print(f"  Recursive(500,50)  块数: {len(chunks_r_500)}  平均长度: {np.mean([len(c) for c in chunks_r_500]):.0f}")
    print(f"  Recursive(1000,100) 块数: {len(chunks_r_1000)} 平均长度: {np.mean([len(c) for c in chunks_r_1000]):.0f}")
    print(f"  Recursive(1500,150) 块数: {len(chunks_r_1500)} 平均长度: {np.mean([len(c) for c in chunks_r_1500]):.0f}")
    print(f"  Semantic(0.5)      块数: {len(chunks_sem)}     平均长度: {np.mean([len(c) for c in chunks_sem]):.0f}")
