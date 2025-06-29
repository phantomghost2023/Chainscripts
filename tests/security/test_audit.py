import pytest
from chainscript.security import audit_log
import os


@pytest.fixture(autouse=True)
def clear_audit_log_file():
    log_file = "audit.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    yield
    if os.path.exists(log_file):
        os.remove(log_file)


def test_audit_log_entry():
    message = "User login successful"
    audit_log(message)
    with open("audit.log", "r") as f:
        content = f.read()
        assert message in content
        assert "INFO" in content


def test_audit_log_multiple_entries():
    message1 = "Attempted unauthorized access"
    message2 = "System configuration changed"
    audit_log(message1)
    audit_log(message2)
    with open("audit.log", "r") as f:
        content = f.read()
        assert message1 in content
        assert message2 in content


def test_audit_log_different_levels():
    audit_log("Warning message", level="WARNING")
    audit_log("Error message", level="ERROR")
    with open("audit.log", "r") as f:
        content = f.read()
        assert "WARNING" in content
        assert "Error message" in content
