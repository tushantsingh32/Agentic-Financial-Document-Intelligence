from __future__ import annotations

import ast
import operator as op
from typing import Any

from langchain_core.tools import tool


_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: op.pos, ast.USub: op.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent too large")
        return float(_ALLOWED_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return float(_ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand)))
    raise ValueError("Only basic arithmetic is allowed")


@tool
def calculator(expression: str) -> str:
    """Safely calculate a basic arithmetic expression such as '(150-120)/120*100'."""
    try:
        tree = ast.parse(expression, mode="eval")
        value = _safe_eval(tree)
        return f"{value:.6g}"
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def percentage_change(old_value: float, new_value: float) -> str:
    """Calculate percentage change from old_value to new_value."""
    if old_value == 0:
        return "Percentage change is undefined when the old value is zero."
    result = (new_value - old_value) / old_value * 100
    return f"{result:.2f}%"


@tool
def profit_margin(revenue: float, profit: float) -> str:
    """Calculate profit margin as a percentage of revenue."""
    if revenue == 0:
        return "Profit margin is undefined when revenue is zero."
    return f"{profit / revenue * 100:.2f}%"


@tool
def cagr(start_value: float, end_value: float, years: float) -> str:
    """Calculate CAGR percentage for positive start/end values."""
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return "CAGR requires positive values and years greater than zero."
    return f"{((end_value / start_value) ** (1 / years) - 1) * 100:.2f}%"


FINANCIAL_TOOLS = [calculator, percentage_change, profit_margin, cagr]
