"""
observability/tracing.py
========================
轻量可观测性：用 LangChain 回调采集「每个节点/每次 LLM/每次工具调用」的
延迟、Token 与成本，落到本地 JSONL，并输出聚合摘要。

设计取舍（面试可讲）：
  - 默认零外部依赖：不装 Langfuse 也能跑，数据落 logs/traces/*.jsonl，便于 CI 与离线分析；
  - Langfuse 作为可选增强：设置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 且安装 langfuse
    后自动附加 handler，实现 trace 可视化（无需改动业务代码）；
  - 成本按 Token 数 × 单价估算，单价可用环境变量覆盖，避免把「估算」当成「账单」。
"""
import json
import os
import statistics
import time
from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler

# 近似单价（元 / 1K tokens），默认按 deepseek-chat 量级；可用环境变量覆盖
_DEFAULT_PRICE = {"input": 0.002, "output": 0.008}


def _price(kind: str) -> float:
    env = os.getenv(f"LLM_PRICE_{kind.upper()}_PER_1K")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return _DEFAULT_PRICE[kind]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return round(ordered[idx], 2)


def _extract_usage(response) -> dict:
    """从 LLM 响应里抽取 (input, output) token 数，兼容多种字段。"""
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if usage:
        return {
            "input": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            "output": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        }
    try:
        message = response.generations[0][0].message
        meta = getattr(message, "usage_metadata", None) or {}
        return {"input": int(meta.get("input_tokens") or 0), "output": int(meta.get("output_tokens") or 0)}
    except (AttributeError, IndexError, TypeError):
        return {"input": 0, "output": 0}


class TraceRecorder(BaseCallbackHandler):
    """采集节点延迟、LLM Token/成本、工具调用次数的回调。"""

    def __init__(self, thread_id: str = "", verbose: bool = False) -> None:
        super().__init__()
        self.thread_id = thread_id
        self.verbose = verbose
        self.events: list[dict] = []
        self._chain_start: dict = {}
        self._node_latency: dict[str, list[float]] = defaultdict(list)
        self._tool_counts: dict[str, int] = defaultdict(int)
        self._llm_calls = 0
        self._tokens = {"input": 0, "output": 0}

    # ---------- 链（图节点） ----------
    def on_chain_start(self, serialized, inputs, *, run_id=None, **kwargs) -> None:
        name = (serialized or {}).get("name") if isinstance(serialized, dict) else None
        self._chain_start[str(run_id)] = (name or "chain", time.perf_counter())

    def on_chain_end(self, outputs, *, run_id=None, **kwargs) -> None:
        started = self._chain_start.pop(str(run_id), None)
        if not started:
            return
        name, t0 = started
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        self._node_latency[name].append(elapsed)
        self.events.append({"kind": "node", "name": name, "latency_ms": elapsed})

    # ---------- LLM ----------
    def on_llm_end(self, response, *, run_id=None, **kwargs) -> None:
        usage = _extract_usage(response)
        self._llm_calls += 1
        self._tokens["input"] += usage["input"]
        self._tokens["output"] += usage["output"]
        self.events.append({"kind": "llm", **usage})

    # ---------- 工具 ----------
    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs) -> None:
        name = (serialized or {}).get("name") if isinstance(serialized, dict) else None
        name = name or "tool"
        self._tool_counts[name] += 1
        self.events.append({"kind": "tool", "name": name})

    # ---------- 聚合 ----------
    def summary(self) -> dict:
        input_tokens = self._tokens["input"]
        output_tokens = self._tokens["output"]
        cost = round(
            input_tokens / 1000 * _price("input") + output_tokens / 1000 * _price("output"), 4
        )
        all_latencies = [v for values in self._node_latency.values() for v in values]
        return {
            "thread_id": self.thread_id,
            "llm_calls": self._llm_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_yuan": cost,
            "tool_counts": dict(self._tool_counts),
            "node_latency_ms": {
                name: {
                    "count": len(values),
                    "p50": _percentile(values, 50),
                    "p95": _percentile(values, 95),
                }
                for name, values in self._node_latency.items()
            },
            "overall_latency_ms": {
                "count": len(all_latencies),
                "p50": _percentile(all_latencies, 50),
                "p95": _percentile(all_latencies, 95),
            },
        }

    def dump(self, directory: str) -> str:
        """把事件与摘要写入 directory/<thread_id>.jsonl，返回文件路径。"""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.thread_id or 'trace'}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "summary", **self.summary()}, ensure_ascii=False) + "\n")
            for event in self.events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return path


def _try_langfuse_handler():
    """若配置了 Langfuse 且已安装，返回其 CallbackHandler，否则返回 None。"""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse.langchain import CallbackHandler  # 新版 langfuse v3+
    except Exception:
        try:
            from langfuse.callback import CallbackHandler  # 旧版兼容
        except Exception:
            return None
    try:
        return CallbackHandler()
    except Exception:
        return None


def build_callbacks(thread_id: str = "", verbose: bool = False) -> list:
    """返回可传给 graph.astream(config={"callbacks": ...}) 的回调列表。"""
    handlers: list = [TraceRecorder(thread_id=thread_id, verbose=verbose)]
    langfuse_handler = _try_langfuse_handler()
    if langfuse_handler is not None:
        handlers.append(langfuse_handler)
    return handlers
