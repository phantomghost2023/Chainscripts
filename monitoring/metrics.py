from prometheus_client import Counter, Histogram, Gauge

EXECUTION_TIME = Histogram(
    "chainscript_execution_seconds", "Script execution time", ["script_type"]
)

HEALING_ATTEMPTS = Counter(
    "chainscript_healing_attempts_total", "Total healing interventions"
)
