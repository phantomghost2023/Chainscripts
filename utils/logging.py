import logging
from datetime import datetime
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from config.settings import ChainScriptConfig


def setup_sentry():
    sentry_logging = LoggingIntegration(
        level=logging.INFO, event_level=logging.ERROR
    )

    sentry_sdk.init(
        dsn=ChainScriptConfig.SENTRY_DSN,
        integrations=[sentry_logging],
        traces_sample_rate=1.0,
    )


class ExecutionLogger:
    def __init__(self):
        self.logger = logging.getLogger("chainscript")
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            filename=f"logs/execution_{datetime.now():%Y%m%d}.log",
            format="%(asctime)s | %(levelname)s | %(message)s",
            level=logging.INFO,
        )

    def log_healing(self, original: str, corrected: str):
        self.logger.info(
            f"Healing applied: {original}→{corrected}",
            extra={"type": "profile_correction"},
        )


# Global logger instance
logger = ExecutionLogger()

# Setup Sentry when the module is imported
setup_sentry()
