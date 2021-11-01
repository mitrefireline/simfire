from time import time
from functools import wraps


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
        print(f'{func.__name__} executed in {end - start:.4f} seconds')
        return result

    return wrapper
