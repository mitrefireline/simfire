import logging
import os
import sys

from rich.logging import RichHandler


def create_logger(name: str) -> logging.Logger:
    """Create a `Logger` to be used in different modules

    Parameters:
    -----------
    name: str
        The name of the logger. Will usually be passed in from the module as `__name__`.

    Returns:
    --------
    log: logging.Logger
        The `Logger` object that will be used to create log statements in the terminal.
    """
    if (log_level := os.environ.get("LOGLEVEL")) is None:
        log_level = "INFO"

    # Not really sure why this worked to suppress log output from all modules besides
    # simfire, but it works so I won't question it too much. - mdoyle
    stdout_logger = logging.getLogger("STDOUT").addHandler(RichHandler())  # type: ignore
    stderr_logger = logging.getLogger("STDERR").addHandler(RichHandler())  # type: ignore

    sys.stdout = stdout_logger
    sys.stderr = stderr_logger

    FORMAT = "%(message)s"
    handlers = [RichHandler()]
    logging.basicConfig(
        level=log_level,
        format=FORMAT,
        datefmt="[%m/%d/%Y %I:%M:%S %p]",
        handlers=handlers,
    )

    log = logging.getLogger(name)
    log.setLevel(log_level)
    return log
