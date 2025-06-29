import pytest
from ..core.nano_engine import execute_script

# Define your maximum expected times and memory usage for each script type
# These should be determined based on your performance benchmarks
MAX_TIMES = {
    "small": 0.1,  # Example value
    "medium": 0.5, # Example value
    "large": 2.0   # Example value
}

MAX_MEMORY = {
    "small": 10,   # Example value in MB
    "medium": 50,  # Example value in MB
    "large": 200   # Example value in MB
}

@pytest.mark.parametrize("script_type", ["small", "medium", "large"])
def test_against_golden(golden_scripts, script_type):
    for script in golden_scripts.get(script_type, []):
        time, mem = execute_script(script)
        assert time < MAX_TIMES[script_type]
        assert mem < MAX_MEMORY[script_type]