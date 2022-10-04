import logging
import os
import sys
from typing import Union 

from rich.logging import RichHandler


class LoggerWriter:
    """Taken from https://docs.python.org/3.11/howto/logging-cookbook.html
    """
    def __init__(self, logger: logging.Logger, level: Union[int, str]) -> None:
        self.logger = logger
        self.level = level

    def write(self, message: str) -> None:
        if message != '\n':  # avoid printing bare newlines, if you like
            self.logger.log(self.level, message)

    def flush(self) -> None:
        # doesn't actually do anything, but might be expected of a file-like
        # object - so optional depending on your situation
        pass

    def close(self) -> None:
        # doesn't actually do anything, but might be expected of a file-like
        # object - so optional depending on your situation. You might want
        # to set a flag so that later calls to write raise an exception
        pass


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
