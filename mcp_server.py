"""MCP server exposing the financial calculation tools.

Run locally with:
    python mcp_server.py
Or with the MCP CLI:
    mcp dev mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Financial Document Analyst Tools")


@mcp.tool()
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression."""
    import ast
    import operator as op

    allowed = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow}
    unary = {ast.UAdd: op.pos, ast.USub: op.neg}

    def ev(node):
        if isinstance(node, ast.Expression): return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(b) > 10: raise ValueError("Exponent too large")
            return allowed[type(node.op)](a, b)
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary: return unary[type(node.op)](ev(node.operand))
        raise ValueError("Only basic arithmetic is allowed")

    try:
        return f"{ev(ast.parse(expression, mode='eval')):.6g}"
    except Exception as exc:
        return f"Calculation error: {exc}"


@mcp.tool()
def percentage_change(old_value: float, new_value: float) -> str:
    """Calculate percentage change from old to new value."""
    if old_value == 0:
        return "Undefined for an old value of zero."
    return f"{(new_value-old_value)/old_value*100:.2f}%"


@mcp.tool()
def profit_margin(revenue: float, profit: float) -> str:
    """Calculate profit margin percentage."""
    if revenue == 0:
        return "Undefined for zero revenue."
    return f"{profit/revenue*100:.2f}%"


@mcp.tool()
def cagr(start_value: float, end_value: float, years: float) -> str:
    """Calculate CAGR percentage."""
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return "Requires positive values and years greater than zero."
    return f"{((end_value/start_value)**(1/years)-1)*100:.2f}%"


if __name__ == "__main__":
    mcp.run()
