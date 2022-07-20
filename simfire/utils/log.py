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

    FORMAT = "%(asctime)s : %(levelname)s : (%(pathname)s:%(lineno)d) : %(message)s"

    logging.basicConfig(
        level=log_level,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(markup=True)],
    )

    log = logging.getLogger(name)
    return log
