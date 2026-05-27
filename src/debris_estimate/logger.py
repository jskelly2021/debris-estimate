"""Logging utilities."""

import logging


RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"


def setup_logger():
    """Configure process-wide console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


class Log:
    def __init__(self):
        self.log = logging.getLogger("MultiLabelModel")

    def info(self, text: str, *args) -> None:
        self.log.info(f"{GREEN}[INFO]{RESET} {text}", *args)

    def warn(self, text: str, *args) -> None:
        self.log.warning(f"{YELLOW}[WARN]{RESET} {text}", *args)

    def error(self, text: str, *args) -> None:
        self.log.error(f"{RED}[ERROR]{RESET} {text}", *args)
