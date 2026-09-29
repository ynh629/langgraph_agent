# exceptions.py
"""
RAG 系统统一异常类模块。
定义系统内各环节的自定义异常，便于顶层统一捕获与处理。
"""
from typing import Optional


class RAGException(Exception):
    """RAG 系统基础异常，所有自定义异常的基类。"""

    def __init__(self, message: str = "", *, code: Optional[str] = None, detail: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.code = code          # 机器可读的错误码，如 "RETRIEVAL_EMPTY"
        self.detail = detail      # 附加的详细说明（如堆栈、上下文）

    def to_dict(self) -> dict:
        """转换为可序列化的字典，便于返回给调用方。"""
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }

    def __str__(self) -> str:
        parts = [self.__class__.__name__]
        if self.code:
            parts.append(f"[{self.code}]")
        parts.append(self.message)
        return " ".join(parts)


class DocumentError(RAGException):
    """文档解析/加载相关异常（PDF 损坏、JSON 格式错误、文件不存在等）。"""


class IndexError_(RAGException):
    """索引构建相关异常（空文档、Chroma 锁、embedding 失败等）。"""


class RetrievalError(RAGException):
    """检索环节异常（向量库查询失败、融合计算异常等）。"""


class RerankError(RAGException):
    """重排序环节异常（模型加载失败、API 调用失败、超长文本等）。"""


class LLMError(RAGException):
    """LLM 生成环节异常（API 超时、限流、鉴权失败、上下文超限等）。"""


class ConfigError(RAGException):
    """配置相关异常（缺少 API Key、环境变量缺失等）。"""
