from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

provider = TracerProvider()
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("chainscript")

def trace_execution(script_name: str):
    return tracer.start_as_current_span(script_name)