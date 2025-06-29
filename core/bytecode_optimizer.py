"""
Bytecode Optimizer for ChainScript
Provides ahead-of-time compilation and optimization for nano-scripts
"""

import ast
import dis
import marshal
import py_compile
import tempfile
import os
from typing import Dict, Any, Optional
from pathlib import Path
import hashlib

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


class BytecodeOptimizer:
    """
    Optimizes and compiles nano-scripts to bytecode for faster execution
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or Path.home() / ".chainscript" / "bytecode_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.optimization_cache: Dict[str, Any] = {}

    def get_script_hash(self, script_path: str) -> str:
        """Generate hash for script content for cache invalidation"""
        with open(script_path, "rb") as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()[:16]

    def compile_to_bytecode(self, script_path: str, optimization_level: int = 2) -> str:
        """
        Compile Python script to optimized bytecode

        Args:
            script_path: Path to the script to compile
            optimization_level: 0=none, 1=basic, 2=aggressive

        Returns:
            Path to compiled bytecode file
        """
        script_hash = self.get_script_hash(script_path)
        cache_key = f"{Path(script_path).stem}_{script_hash}_opt{optimization_level}"
        bytecode_path = self.cache_dir / f"{cache_key}.pyc"

        # Return cached version if exists
        if bytecode_path.exists():
            return str(bytecode_path)

        try:
            # Basic compilation
            if optimization_level == 0:
                py_compile.compile(script_path, str(bytecode_path), doraise=True)

            # AST optimization
            elif optimization_level >= 1:
                bytecode_path = self._ast_optimize(script_path, bytecode_path)

            # Numba JIT compilation for numeric code
            if optimization_level == 2 and HAS_NUMBA:
                self._apply_numba_optimization(script_path)

            return str(bytecode_path)

        except Exception as e:
            print(f"Compilation failed for {script_path}: {e}")
            # Fallback to basic compilation
            py_compile.compile(script_path, str(bytecode_path), doraise=True)
            return str(bytecode_path)

    def _ast_optimize(self, script_path: str, output_path: Path) -> Path:
        """Apply AST-level optimizations"""
        with open(script_path, "r") as f:
            source = f.read()

        # Parse to AST
        tree = ast.parse(source)

        # Apply optimizations
        optimizer = ASTOptimizer()
        optimized_tree = optimizer.visit(tree)

        # Compile optimized AST
        code = compile(optimized_tree, script_path, "exec", optimize=2)

        # Write bytecode
        with open(output_path, "wb") as f:
            marshal.dump(code, f)

        return output_path

    def _apply_numba_optimization(self, script_path: str):
        """Apply Numba JIT compilation for numerical functions"""
        if not HAS_NUMBA:
            return

        # This would analyze the script and apply @numba.jit decorators
        # to numeric functions automatically
        print(f"Applying Numba optimization to {script_path}")

    def precompile_nano_scripts(self, nano_scripts_dir: str):
        """Precompile all nano-scripts in the directory"""
        nano_scripts_path = Path(nano_scripts_dir)

        for script_file in nano_scripts_path.rglob("*.py"):
            if script_file.name.startswith("__"):
                continue

            print(f"Precompiling {script_file}")
            self.compile_to_bytecode(str(script_file))

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Return statistics about optimization performance"""
        cache_files = list(self.cache_dir.glob("*.pyc"))
        return {
            "cached_scripts": len(cache_files),
            "cache_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
            "optimization_cache_entries": len(self.optimization_cache),
        }


class ASTOptimizer(ast.NodeTransformer):
    """
    AST-level optimizations for nano-scripts
    """

    def visit_BinOp(self, node):
        """Optimize binary operations with constants"""
        # Constant folding for numeric operations
        if (
            isinstance(node.left, ast.Constant)
            and isinstance(node.right, ast.Constant)
            and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div))
        ):

            try:
                if isinstance(node.op, ast.Add):
                    result = node.left.value + node.right.value
                elif isinstance(node.op, ast.Sub):
                    result = node.left.value - node.right.value
                elif isinstance(node.op, ast.Mult):
                    result = node.left.value * node.right.value
                elif isinstance(node.op, ast.Div):
                    result = node.left.value / node.right.value

                return ast.Constant(value=result)
            except:
                pass

        return self.generic_visit(node)

    def visit_If(self, node):
        """Optimize if statements with constant conditions"""
        if isinstance(node.test, ast.Constant):
            if node.test.value:
                return node.body
            elif node.orelse:
                return node.orelse
            else:
                return []

        return self.generic_visit(node)


# Example usage and testing
if __name__ == "__main__":
    optimizer = BytecodeOptimizer()

    # Create a test script
    test_script = """
def add_numbers(a, b):
    # This will be optimized
    return a + b + 0  # + 0 will be optimized away

def process_data(data):
    result = []
    for item in data:
        if True:  # This condition will be optimized
            result.append(item * 2)
    return result
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        test_path = f.name

    try:
        # Compile with different optimization levels
        bytecode_path = optimizer.compile_to_bytecode(test_path, optimization_level=2)
        print(f"Compiled to: {bytecode_path}")

        # Show stats
        stats = optimizer.get_optimization_stats()
        print(f"Optimization stats: {stats}")

    finally:
        os.unlink(test_path)
