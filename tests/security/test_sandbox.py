import pytest
from chainscript.security import execute_in_sandbox


def test_execute_in_sandbox_safe_code():
    code = "result = 1 + 1"
    result = execute_in_sandbox(code)
    assert result == 2


def test_execute_in_sandbox_unsafe_code_file_access():
    code = "import os; result = os.listdir('.')"
    with pytest.raises(Exception):
        execute_in_sandbox(code)


def test_execute_in_sandbox_unsafe_code_system_call():
    code = "import subprocess; result = subprocess.run(['ls'])"
    with pytest.raises(Exception):
        execute_in_sandbox(code)


def test_execute_in_sandbox_timeout():
    code = "import time; time.sleep(2)"
    with pytest.raises(Exception):
        execute_in_sandbox(code, timeout=1)


def test_execute_in_sandbox_return_value():
    code = "def func(): return 'hello'; result = func()"
    result = execute_in_sandbox(code)
    assert result == "hello"


def test_execute_in_sandbox_syntax_error():
    code = "result = 1 ++"
    with pytest.raises(Exception):
        execute_in_sandbox(code)
