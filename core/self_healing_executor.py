from dataclasses import dataclass
import time
from .auto_classifier import AutoClassifier, ScriptSignature # Assuming auto_classifier.py is in the same 'core' directory

@dataclass
class ExecutionResult:
    """Represents the result of a script execution, including performance metrics."""
    exec_time: float
    mem_usage: float

class HealingExecutor:
    """Executes scripts with adaptive healing capabilities based on performance."""
    def __init__(self, detect_profile_func, get_strategy_func, execute_script_func, log_correction_func, auto_classification_error_class):
        self.classifier = AutoClassifier()
        self.performance_log = []
        self._detect_profile = detect_profile_func
        self._get_strategy = get_strategy_func
        self._execute_script = execute_script_func
        self._log_correction = log_correction_func
        self._AutoClassificationError = auto_classification_error_class
    
    def execute_with_healing(self, script_path: str):
        """Executes a script and attempts to heal performance issues adaptively."""
        signature = self.classifier.analyze_script(script_path)
        initial_profile_info = self._detect_profile(script_path)
        initial_profile_category = initial_profile_info.get('size_category', 'default')
        
        for attempt in range(3):  # Max 3 correction attempts 
            strategy = self._get_strategy({'size_category': initial_profile_category}) 
            result = self._execute_script(script_path, strategy) 
            
            if self._is_performance_acceptable(result, initial_profile_category): 
                return result 
            
            new_profile_category = self.classifier.suggest_fix( 
                signature, 
                {"time": result.exec_time, "memory": result.mem_usage} 
            ) 
            self._log_correction(initial_profile_category, new_profile_category) 
            initial_profile_category = new_profile_category 
        
        raise self._AutoClassificationError("Max corrections reached") 

    def _is_performance_acceptable(self, result, profile_category): 
        golden = self.classifier.golden_benchmarks.get(profile_category, {}) 
        return (result.exec_time <= golden.get("max_time", float('inf')) and \
               (result.mem_usage <= golden.get("max_mem", float('inf'))))