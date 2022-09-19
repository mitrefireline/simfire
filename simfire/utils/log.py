import logging
import os

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

    FORMAT = "%(message)s"
    handlers = [RichHandler(markup=False)]
    logging.basicConfig(
        level=log_level,
        format=FORMAT,
        datefmt="[%m/%d/%Y %I:%M:%S %p]",
        handlers=handlers,
    )

    log = logging.getLogger(name)
    log.setLevel(log_level)
    return log
