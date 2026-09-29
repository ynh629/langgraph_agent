# logging_config.py
"""
统一日志配置模块。
为 RAG 系统所有模块提供统一的日志记录能力：
- 控制台 + 文件双输出
- 文件使用 RotatingFileHandler（单文件最大 10MB，保留 3 个备份）
- 通过环境变量 LOG_LEVEL 控制日志级别（默认 INFO）
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# rag_system 包根目录（common 的上一级）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 日志文件目录（位于 rag_system/logs）
LOG_DIR = os.path.join(BASE_DIR, "logs")
# 日志文件路径
LOG_FILE = os.path.join(LOG_DIR, "rag_system.log")

# 日志格式：时间 | 级别 | 模块:行号 | 消息
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 已初始化的 logger 缓存，避免重复添加 handler
_loggers = {}


def _ensure_log_dir():
    """确保日志目录存在。"""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except OSError as e:
        # 目录创建失败时退化为仅控制台输出
        print(f"[日志] 无法创建日志目录 {LOG_DIR}: {e}", file=sys.stderr)


def get_logger(name: str = "rag_system") -> logging.Logger:
    """
    获取（或创建）一个统一配置的 logger。

    参数：
        name: logger 名称，通常传入 __name__（模块名）

    返回：
        配置好的 logging.Logger 实例
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(_resolve_level())

    # 避免重复添加 handler（防止多次调用时重复输出）
    if logger.handlers:
        _loggers[name] = logger
        return logger

    # 1. 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(_resolve_level())
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(console_handler)

    # 2. 文件 handler（RotatingFileHandler）
    try:
        _ensure_log_dir()
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,              # 保留 3 个备份
            encoding="utf-8"
        )
        file_handler.setLevel(_resolve_level())
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        # 文件 handler 失败不影响控制台输出
        print(f"[日志] 无法创建文件 handler: {e}", file=sys.stderr)

    # 防止日志向上传播到 root logger 造成重复
    logger.propagate = False

    _loggers[name] = logger
    return logger


def _resolve_level() -> int:
    """根据环境变量 LOG_LEVEL 解析日志级别。"""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    env_level = os.getenv("LOG_LEVEL", "INFO").upper()
    return level_map.get(env_level, logging.INFO)


# 模块级便捷引用
logger = get_logger(__name__)
