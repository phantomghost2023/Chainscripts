import ast
from dataclasses import dataclass
import numpy as np
import ast

@dataclass
class ScriptSignature:
    """Represents the analyzed signature of a script."""
    loc: int
    complexity: float  # 0-1 (simple-complex)
    imports: list
    loop_depth: int

class AutoClassifier:
    """Analyzes script characteristics and suggests execution strategies."""
    def __init__(self) -> None:
        self.golden_benchmarks = {
            "small": {"max_time": 0.001, "max_mem": 50},
            "medium-lightweight": {"max_time": 0.01, "max_mem": 100},
            "large": {"max_time": 0.1, "max_mem": 200}
        }
    
    def analyze_script(self, path: str) -> ScriptSignature:
        """
        Analyzes a Python script file to extract its signature.

        Args:
            path: The file path to the script.

        Returns:
            A ScriptSignature object containing the script's characteristics.
        """
        with open(path) as f:
            tree = ast.parse(f.read())
        
        return ScriptSignature(
            loc=len(tree.body),
            complexity=self._calc_complexity(tree),
            imports=self._extract_imports(tree),
            loop_depth=self._max_loop_depth(tree)
        )

    def _calc_complexity(self, tree: ast.AST) -> float:
        """
        Calculates a complexity score for the given AST.

        Args:
            tree: The AST (Abstract Syntax Tree) of the script.

        Returns:
            A float representing the complexity score, normalized to a 0-1 range.
        """
        # Simple complexity score based on AST nodes
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                complexity += 5
            elif isinstance(node, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
                complexity += 3
            elif isinstance(node, (ast.Call, ast.Compare)):
                complexity += 1
        # Normalize to 0-1 range (example normalization, adjust as needed)
        return min(1.0, complexity / 100.0) 

    def _extract_imports(self, tree: ast.AST) -> list[str]:
        """
        Extracts import names from the AST.

        Args:
            tree: The AST (Abstract Syntax Tree) of the script.

        Returns:
            A list of strings, where each string is the name of an imported module.
        """
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports

    def _max_loop_depth(self, tree: ast.AST) -> int:
        """
        Calculates the maximum nested loop depth within the AST.

        Args:
            tree: The AST (Abstract Syntax Tree) of the script.

        Returns:
            An integer representing the maximum loop depth.
        """
        # Calculate maximum nested loop depth
        max_depth = 0
        current_depth = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            # This simple approach doesn't decrement depth for exiting a loop block
            # A more robust solution would require a visitor pattern or tracking parent nodes
        return max_depth
    
    def suggest_fix(self, script: ScriptSignature, actual_perf: dict) -> str:
        """
        Suggests a script profile based on its signature and actual performance.

        Args:
            script: The ScriptSignature object of the script.
            actual_perf: A dictionary containing 'time' and 'memory' performance metrics.

        Returns:
            A string representing the suggested script profile (e.g., "small", "medium-lightweight").
        """
        # Compare against golden benchmarks 
        for profile, benchmarks in self.golden_benchmarks.items(): 
            if (actual_perf["time"] <= benchmarks["max_time"] and 
                actual_perf["memory"] <= benchmarks["max_mem"]): 
                return profile  # First matching profile 
        
        # Emergency fallback rules 
        if script.loc < 50: return "small" 
        if "pandas" in script.imports: return "large" 
        return "medium-balanced"