import restrictedpython

def safe_eval(code: str):
    """Executes code in a restricted environment"""
    return restrictedpython.compile_restricted(
        code,
        filename="<string>",
        mode="eval"
    )