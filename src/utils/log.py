import logging
import os


def create_logger(name: str) -> logging.Logger:
    '''Create a `Logger` to be used in different modules

    Parameters:
    -----------
    name: str
        The name of the logger. Will usually be passed in from the module as `__name__`.

    Returns:
    --------
    log: logging.Logger
        The `Logger` object that will be used to create log statements in the terminal.
    '''
    if (log_level := os.environ.get('LOGLEVEL')) is None:
        log_level = 'INFO'

    formatter = logging.Formatter('%(asctime)s : %(levelname)s : '
                                  '(%(pathname)s:%(lineno)d) : %(message)s')

    log = logging.getLogger(name)
    log.setLevel(log_level)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    log.addHandler(console)
    return log
