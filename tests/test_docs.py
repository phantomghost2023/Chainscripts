import ast
import pathlib
import pytest

def test_all_functions_have_docstrings():
    missing = []
    for path in pathlib.Path("chainscript/core").rglob("*.py"):
        # Skip __init__.py files as they often don't require docstrings
        if path.name == "__init__.py":
            continue
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Exclude private methods/functions (starting with _)
                if node.name.startswith('_'):
                    continue
                if not ast.get_docstring(node):
                    missing.append(f"{path}:{node.lineno} {node.name}")
    
    assert not missing, f"Missing docstrings:\n{chr(10).join(missing)}"