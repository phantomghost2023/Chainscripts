import ast
import os
from typing import Tuple, Optional
from pathlib import Path
from .self_healing_executor import HealingExecutor, ExecutionResult


class AutoClassificationError(Exception):
    """Custom exception for auto-classification errors."""


class NanoScriptEngine:
    """A script engine for analyzing, compiling, and executing NanoScripts."""

    def __init__(self):
        self.cache = {}
        self.script_profiles = (
            {}
        )  # Store script characteristics for adaptive optimization

        # Initialize HealingExecutor with necessary callbacks
        self.healing_executor = HealingExecutor(
            detect_profile_func=self.analyze_script_complexity,
            get_strategy_func=self._get_execution_strategy,
            execute_script_func=self._execute_script_internal,
            log_correction_func=self._log_correction,
            auto_classification_error_class=AutoClassificationError,
        )

        # Execution mode thresholds (adjusted for predictable simulated ranges)
        self.SMALL_SCRIPT_THRESHOLD = 85  # lines of code (0-85: small scripts)
        self.MEDIUM_SCRIPT_THRESHOLD = (
            150  # lines of code (86-150: medium-lightweight scripts)
        )
        self.LARGE_SCRIPT_THRESHOLD = (
            300  # lines of code (151-300: medium-balanced, >300: large)
        )

    def analyze_script_complexity(self, script_path: Path):
        """Analyze script to determine optimal execution strategy"""
        try:
            if script_path.exists():
                with open(script_path, "r") as f:
                    content = f.read()

                # Count lines of actual code (excluding comments and empty lines)
                lines = [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip() and not line.strip().startswith("#")
                ]
                line_count = len(lines)

                # Parse AST to count function definitions and loops
                try:
                    tree = ast.parse(content)
                    complexity_score = self._calculate_ast_complexity(tree)
                except Exception as e:
                    print(f"Warning: AST parsing failed: {e}")
                    complexity_score = line_count  # Fallback to line count

                return {
                    "line_count": line_count,
                    "complexity_score": complexity_score,
                    "size_category": self._categorize_script_size(line_count),
                }
            else:
                # For demo purposes, simulate script analysis with predictable categories
                simulated_lines = self._simulate_realistic_script_size(script_path)
                return {
                    "line_count": simulated_lines,
                    "complexity_score": simulated_lines * 1.2,
                    "size_category": self._categorize_script_size(simulated_lines),
                }
        except Exception as e:
            print(f"Warning: Could not analyze script {script_path}: {e}")
            return {
                "line_count": 100,
                "complexity_score": 120,
                "size_category": "medium",
            }

    def _calculate_ast_complexity(self, tree):
        """Calculate complexity score based on AST analysis"""
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity += 10
            elif isinstance(node, (ast.For, ast.While, ast.If)):
                complexity += 5
            elif isinstance(node, (ast.Try, ast.With)):
                complexity += 3
            elif isinstance(node, ast.ClassDef):
                complexity += 15
        return complexity

    def _simulate_realistic_script_size(self, script_path: Path):
        """Simulate realistic script sizes based on script name patterns"""
        script_name = os.path.basename(script_path).lower()

        # Assign sizes based on typical script patterns
        if any(
            keyword in script_name for keyword in ["small", "tiny", "micro", "quick"]
        ):
            base_size = 25
        elif any(
            keyword in script_name for keyword in ["medium", "standard", "regular"]
        ):
            base_size = 100
        elif any(
            keyword in script_name for keyword in ["large", "big", "complex", "heavy"]
        ):
            base_size = 200
        else:
            # Use script name characteristics to determine realistic size
            name_hash = hash(script_name)
            if "test" in script_name or "demo" in script_name:
                base_size = 30 + (name_hash % 40)  # 30-70 lines (small)
            elif "api" in script_name or "fetch" in script_name:
                base_size = 60 + (name_hash % 60)  # 60-120 lines (small to medium)
            elif "clean" in script_name or "process" in script_name:
                base_size = 80 + (name_hash % 80)  # 80-160 lines (medium)
            elif "analyze" in script_name or "transform" in script_name:
                base_size = 120 + (name_hash % 100)  # 120-220 lines (medium to large)
            else:
                base_size = 50 + (name_hash % 100)  # 50-150 lines (varied)

        # Add some controlled variation
        variation = (hash(script_path) % 20) - 10  # ±10 lines variation
        return max(5, base_size + variation)  # Minimum 5 lines

    def _categorize_script_size(self, line_count):
        """Categorize script size for execution strategy selection"""
        if line_count < 50:
            return "small"
        elif line_count < 100:
            return "medium-lightweight"
        elif line_count < 200:
            return "medium-balanced"
        else:
            return "large"

    def load_script(self, script_name: str):
        """Loads a script from the given path."""
        # Check if script is cached
        if script_name in self.cache:
            return self.cache[script_name]

        # Analyze script for adaptive optimization
        script_path = Path(f"nano_scripts/{script_name}.py")
        profile = self.analyze_script_complexity(script_path)
        self.script_profiles[script_name] = profile

        # Choose compilation strategy based on script characteristics
        optimized_script = self.adaptive_compile(script_path, profile)

        self.cache[script_name] = optimized_script
        return optimized_script

    def adaptive_compile(self, script_path: Path, profile: dict):
        """Compile script using adaptive strategy based on size and complexity"""
        size_category = profile["size_category"]


        if size_category == "small":
            # For small scripts: Use aggressive optimization with JIT
            return self.compile_with_jit_optimization(script_path)
        elif size_category == "medium-lightweight":
            # For medium-lightweight scripts: Use lightweight optimization
            return self.compile_lightweight(script_path)
        elif size_category == "medium-balanced":
            # For medium-balanced scripts: Use balanced optimization
            return self.compile_balanced(script_path)
        else:
            # For large scripts: Use full optimization suite
            return self.compile_with_full_optimization(script_path)

    def compile_with_jit_optimization(self, script_path: Path):
        """Aggressive optimization for small scripts"""
        print(f"Compiling {script_path} with JIT optimization for small script...")
        return f"jit_optimized_bytecode_of_{script_path}"

    def compile_lightweight(self, script_path: Path):
        """Lightweight compilation for medium scripts to reduce overhead"""
        print(
            f"Compiling {script_path} with lightweight optimization for medium script..."
        )
        return f"lightweight_bytecode_of_{script_path}"

    def compile_balanced(self, script_path: Path):
        """Balanced optimization for complex medium scripts"""
        print(
            f"Compiling {script_path} with balanced optimization for complex medium script..."
        )
        return f"balanced_bytecode_of_{script_path}"

    def compile_with_full_optimization(self, script_path: Path):
        """Full optimization suite for large scripts"""
        print(f"Compiling {script_path} with full optimization for large script...")
        return f"fully_optimized_bytecode_of_{script_path}"

    def compile_to_bytecode(self, script_path: Path):
        """Compiles script content into bytecode."""
        # Legacy method - now redirects to adaptive compilation
        profile = self.analyze_script_complexity(script_path)
        return self.adaptive_compile(script_path, profile)

    def resolve_script_path(self, script_name: str) -> Path:
        """Resolves the full path for a given script name."""
        search_dirs = [
            "nano_scripts",
            "nano_scripts/api",
            "nano_scripts/data",
            "nano_scripts/utils",
        ]

        for dir_str in search_dirs:
            dir_path = Path(dir_str)
            script_full_path = dir_path / f"{script_name}.py"
            if script_full_path.exists():
                return script_full_path

        raise FileNotFoundError(
            f"Script '{script_name}.py' not found in: {', '.join(search_dirs)}"
        )

    def execute_script(
        self, script_name: str, test_data: Optional[dict] = None
    ) -> Tuple[float, float]:
        """
        Executes a script with performance monitoring.

        Args:
            script_name: Name of the script to execute (without .py extension).
            test_data: Optional test inputs (currently unused but for future expansion).

        Returns:
            Tuple of (execution_time_sec, memory_usage_mb) - simulated values.

        Raises:
            FileNotFoundError: If script cannot be located.
            ExecutionError: For runtime failures (simulated).
        """
        script_path: Path = self.resolve_script_path(script_name)
        try:
            result = self.healing_executor.execute_with_healing(script_path)
            print(
                f"Execution completed successfully with healing. Time: {result.exec_time:.4f}s"
            )
            return result.exec_time
        except AutoClassificationError as e:
            print(f"Error: {e}")
            # Fallback to default execution or raise error
            return -1  # Indicate failure

    def _execute_script_internal(
        self, script_path: str, strategy: str
    ) -> ExecutionResult:
        """Internal method to simulate script execution and return performance metrics."""

        # Simulate actual script execution based on strategy
        # For now, we'll use a placeholder for memory usage
        simulated_mem_usage = 10.0  # Placeholder

        # Simulate execution time based on strategy and script characteristics
        # This is a simplified simulation; in a real system, this would run the actual script
        # and measure real time/memory.

        profile = self.analyze_script_complexity(script_path)

        base_time = (
            profile["line_count"] * 0.0001 + profile["complexity_score"] * 0.00005
        )
        if strategy == "fast-jit":
            simulated_exec_time = base_time * 0.5
        elif strategy == "lite":
            simulated_exec_time = base_time * 0.8
        elif strategy == "balanced":
            simulated_exec_time = base_time * 1.0
        elif strategy == "full-optimization":
            simulated_exec_time = base_time * 0.7
        else:
            simulated_exec_time = base_time * 1.2  # Default/unoptimized

        # Add some random noise for more realistic simulation
        simulated_exec_time += time.time() % 0.001

        print(
            f"Simulating execution of {script_path} with {strategy} strategy. Time: {simulated_exec_time:.4f}s"
        )
        return ExecutionResult(
            exec_time=simulated_exec_time, mem_usage=simulated_mem_usage
        )

    def _log_correction(self, original_profile: str, corrected_profile: str):
        """Logs when a script's profile is corrected."""
        print(
            f"[Auto-Correction] Script profile corrected from '{original_profile}' to '{corrected_profile}'"
        )

    def _get_execution_strategy(self, profile):
        """Determine execution strategy based on script profile"""
        if not profile:
            return "default"

        size_category = profile.get(
            "size_category", "medium-balanced"
        )  # Default to a common category

        STRATEGY_MAP = {
            "small": "fast-jit",
            "medium-lightweight": "lite",
            "medium-balanced": "balanced",
            "large": "full-optimization",  # Force full opt for large
        }
        return STRATEGY_MAP.get(size_category, "default")


# Test the NanoScriptEngine
if __name__ == "__main__":
    engine = NanoScriptEngine()
    engine.execute_script("clean_csv", input="data.csv", output="cleaned_data.csv")

