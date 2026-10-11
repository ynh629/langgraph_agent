"""
eval/run_eval.py
================
V4 评估体系统一入口：调度 agent / guardrail / rag / judge 四层评估，
输出「控制台表格 + 结构化 JSON 报告」，并按阈值给出 CI 退出码。

分层设计（面试可讲）：
  - agent     ：脚本化 LLM 驱动图，做【确定性编排门禁】，可在 CI 稳定复现；
  - guardrail ：规则护栏，统计攻击召回率与误拦率，同样确定性；
  - rag       ：检索质量（Hit@k / MRR），依赖 Chroma，环境不可用时标记 skipped；
  - judge     ：LLM-as-judge 主观质检，依赖真实 LLM，默认离线使用，不进 CI 硬门禁。

用法：
    python -m eval.run_eval                        # 默认跑 agent,guardrail（CI 门禁）
    python -m eval.run_eval --layers agent,guardrail,rag
    python -m eval.run_eval --layers judge --no-gate   # 只出报告，不影响退出码
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_AGENT_DIR, os.path.join(_AGENT_DIR, "检索")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from . import agent_eval, guardrail_eval, judge, rag_eval

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

LAYER_ORDER = ("agent", "guardrail", "rag", "judge")
DEFAULT_LAYERS = "agent,guardrail"

# 门禁阈值：确定性层要求满分/高召回，统计层给出容差。
# 每项形如 (指标名, 比较符, 阈值)。
_THRESHOLDS = {
    "agent": [("task_success", ">=", 1.0)],
    "guardrail": [("attack_detection_rate", ">=", 0.9), ("false_positive_rate", "<=", 0.1)],
    "rag": [("hit@3", ">=", 0.8)],
    "judge": [("average_score", ">=", 3.5)],
}

# judge 层的离线抽检样本（真实 LLM 不可用时该层自动 skipped）
_JUDGE_SAMPLES = [
    {
        "id": "j1",
        "question": "订单 SO20260902002 的牛奶变质了，能退款吗？",
        "answer": "已核实订单 SO20260902002，实付 59.9 元，符合签收后 48 小时内变质退款条件，退款 59.9 元已受理，预计 1-3 个工作日原路退回。",
        "context": "订单 SO20260902002 实付 59.9 元，状态已完成。",
    },
    {
        "id": "j2",
        "question": "星耀 X1 打 9 折多少钱？",
        "answer": "星耀 X1 现价 3999 元，9 折后为 3599.1 元。",
        "context": "星耀 X1 售价 3999 元。",
    },
    {
        "id": "j3",
        "question": "帮我查下快递到哪了。",
        "answer": "请提供订单号，我帮您查询最新物流轨迹。",
        "context": "",
    },
]


def _compare(value, op: str, threshold) -> bool:
    if value is None:
        return False
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    raise ValueError(f"不支持的比较符：{op}")


def _evaluate_gate(layer: str, result: dict) -> tuple[bool, list[str]]:
    """按阈值判定某层是否通过门禁，返回 (是否通过, 说明列表)。"""
    if result.get("status") == "skipped":
        return True, [f"{layer}: skipped（{result.get('reason', '依赖不可用')}）"]
    summary = result.get("summary") or {}
    notes, passed = [], True
    for metric, op, threshold in _THRESHOLDS.get(layer, []):
        value = summary.get(metric)
        ok = _compare(value, op, threshold)
        passed = passed and ok
        notes.append(f"{layer}.{metric}={value} {op} {threshold} → {'PASS' if ok else 'FAIL'}")
    if not notes:
        notes.append(f"{layer}: 无阈值，仅记录")
    return passed, notes


def _run_layer(layer: str) -> dict:
    if layer == "agent":
        return asyncio.run(agent_eval.run())
    if layer == "guardrail":
        return guardrail_eval.run()
    if layer == "rag":
        try:
            return rag_eval.run()
        except rag_eval.RetrievalUnavailable as exc:
            return {"layer": "rag", "status": "skipped", "reason": str(exc), "cases": []}
    if layer == "judge":
        return asyncio.run(judge.run(_JUDGE_SAMPLES))
    raise ValueError(f"未知评估层：{layer}")


def _print_layer(result: dict) -> None:
    layer = result["layer"]
    status = result.get("status", "ok")
    summary = result.get("summary") or {}
    head = f"[{layer}] status={status}"
    if summary:
        metrics = "  ".join(f"{k}={v}" for k, v in summary.items())
        head += f"  {metrics}"
    print(head)
    for case in result.get("cases", []):
        if case.get("ok") is False:
            print(f"    ✗ {case.get('id')} {case.get('title', '')}：{'；'.join(case.get('failures', []))}")


def main() -> int:
    parser = argparse.ArgumentParser(description="V4 多层级评估入口")
    parser.add_argument("--layers", default=DEFAULT_LAYERS, help=f"逗号分隔，可选 {LAYER_ORDER}")
    parser.add_argument("--out", default=REPORT_DIR, help="报告输出目录")
    parser.add_argument("--no-gate", action="store_true", help="只出报告，不因阈值失败而返回非零退出码")
    args = parser.parse_args()

    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    unknown = [x for x in layers if x not in LAYER_ORDER]
    if unknown:
        print(f"未知评估层：{unknown}，可选 {LAYER_ORDER}", file=sys.stderr)
        return 2

    print("=" * 72)
    print(f"V4 评估报告  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  层={layers}")
    print("=" * 72)

    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "layers": {}}
    gate_notes: list[str] = []
    all_passed = True

    for layer in layers:
        result = _run_layer(layer)
        report["layers"][layer] = result
        _print_layer(result)
        passed, notes = _evaluate_gate(layer, result)
        all_passed = all_passed and passed
        gate_notes.extend(notes)

    print("-" * 72)
    for note in gate_notes:
        print(f"  {note}")
    report["gate"] = {"passed": all_passed, "notes": gate_notes}
    print("-" * 72)
    print(f"总体门禁：{'PASS ✅' if all_passed else 'FAIL ❌'}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        os.makedirs(args.out, exist_ok=True)
        for name in (f"eval_report_{stamp}.json", "latest.json"):
            with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告已写入：{os.path.join(args.out, f'eval_report_{stamp}.json')}")
    except OSError as exc:
        # 只读/受限文件系统（如受限沙箱）下报告落盘失败，不应影响门禁结论
        print(f"[warn] 报告落盘失败，已跳过：{exc}", file=sys.stderr)

    if args.no_gate:
        return 0
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
