"""
verify_tools_logic.py
=====================
独立验证电商工具核心逻辑（不依赖 langgraph/chromadb 等重依赖）。
用于在无 LLM 环境下，确认链式调用的数据流正确：
  第 1 步 get_product_detail 精确查商品 → 得到 price；
  第 2 步 calculate 对该 price 做运算 → 得到到手价。
同时验证 calculate 对非法表达式（注入）的拦截。

运行：python scripts/verify_tools_logic.py
"""
import ast
import json
import operator
import os
from difflib import SequenceMatcher

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTS_JSON = os.path.join(AGENT_DIR, "data", "ecommerce_products.json")


def load_products():
    with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)["products"]


def get_product_detail(product_name):
    """与 tools.get_product_detail 相同的匹配逻辑。"""
    name = (product_name or "").strip()
    best = None
    best_score = 0.0
    for p in load_products():
        if p["name"] == name:
            score = 1.0
        elif name in p["name"] or p["name"] in name:
            score = 0.9
        else:
            score = SequenceMatcher(None, name, p["name"]).ratio()
        if score > best_score:
            best_score = score
            best = p
    if best is not None and best_score >= 0.5:
        return best
    return None


def safe_eval(expression):
    """与 tools._safe_eval 相同的 AST 白名单求值逻辑。"""
    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](walk(node.operand))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"非法节点: {type(node).__name__}")

    return walk(ast.parse(expression, mode="eval"))


def main():
    print("=" * 60)
    print("验证 1：get_product_detail 精确查商品")
    p = get_product_detail("星耀 X1 手机")
    assert p is not None, "未找到商品"
    assert p["name"] == "星耀 X1 智能手机", f"匹配到错误商品 {p['name']}"
    print(f"  ✓ 找到「{p['name']}」，price={p['price']}")

    print("=" * 60)
    print("验证 2：链式 —— price → calculate(打 9 折)")
    discounted = round(float(safe_eval(f"{p['price']} * 0.9")), 2)
    expected = round(3999 * 0.9, 2)
    assert discounted == expected, f"计算错误 {discounted} != {expected}"
    print(f"  ✓ {p['price']} * 0.9 = {discounted}")

    print("=" * 60)
    print("验证 3：满减叠加计算")
    result = round(float(safe_eval("(3999 * 0.9) - 300")), 2)
    print(f"  ✓ (3999 * 0.9) - 300 = {result}")

    print("=" * 60)
    print("验证 4：安全拦截（防注入）")
    attacks = ["__import__('os').system('dir')", "open('/etc/passwd')", "1 + __builtins__"]
    for expr in attacks:
        try:
            safe_eval(expr)
            print(f"  ✗ 未拦截非法表达式: {expr}")
            raise SystemExit(1)
        except (ValueError, SyntaxError):
            print(f"  ✓ 已拦截: {expr}")

    print("=" * 60)
    print("全部验证通过：链式数据流正确，计算安全。")


if __name__ == "__main__":
    main()