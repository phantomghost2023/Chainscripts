from typing import Dict, Any
import os
import importlib
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

logger = logging.getLogger(__name__)


class ChainScriptConfig:
    SCRIPT_DIRS: Dict[str, Any] = {"nano": "nano_scripts", "golden": "golden_scripts"}
    PROFILE_THRESHOLDS: Dict[str, Any] = {
        "small": {"complexity": 10, "imports": 3},
        "medium": {"complexity": 30, "imports": 7},
        "large": {"complexity": 50, "imports": 12},
    }
    MAX_CORRECTION_ATTEMPTS: int = 3
    FALLBACK_STRATEGY: str = "balanced"
    SENTRY_DSN: str = ""  # Will be configured separately


class ConfigWatcher(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("settings.py"):
            importlib.reload(sys.modules["config.settings"])
            logger.info("Configuration reloaded")


def start_config_watcher():
    observer = Observer()
    observer.schedule(ConfigWatcher(), path="config")
    observer.start()


# Singleton instance
config = ChainScriptConfig()
