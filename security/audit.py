from datetime import datetime


def log_privileged_action(user: str, action: str, script: str):
    with open("/var/log/chainscript/audit.log", "a") as f:
        f.write(f"{datetime.utcnow()} | {user} | {action} | {script}\n")
