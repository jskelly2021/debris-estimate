"""Logging utilities."""

import logging


RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"


def setup_logger(verbose: bool = False):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    logging.getLogger("MultiLabelModel").setLevel(
        logging.DEBUG if verbose else logging.INFO
    )

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


class Log:
    def __init__(self, verbose: bool = True):
        self.log = logging.getLogger("MultiLabelModel")
        self.verbose = verbose

    def info(self, text: str, *args) -> None:
        if self.verbose:
            self.log.info(f"{GREEN}[INFO]{RESET} {text}", *args)

    def debug(self, text: str, *args) -> None:
        self.log.debug(f"{CYAN}[DEBUG]{RESET} {text}", *args)

    def warn(self, text: str, *args) -> None:
        self.log.warning(f"{YELLOW}[WARN]{RESET} {text}", *args)

    def error(self, text: str, *args) -> None:
        self.log.error(f"{RED}[ERROR]{RESET} {text}", *args)
