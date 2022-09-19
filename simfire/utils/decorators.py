from functools import wraps
from time import time

from ..utils.log import create_logger

log = create_logger(__name__)


def timeit(func):
    """
    :param func: Decorated function
    :return: Execution time for the decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        log.info(f"{func.__name__} executed in {end - start:.4f} seconds")
        return result

    return wrapper
