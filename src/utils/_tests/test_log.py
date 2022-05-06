import logging
import os
import unittest

from ..log import create_logger


class LogTest(unittest.TestCase):
    def test_create_logger(self) -> None:
        """
        Test creating a logger with the correct name and log level
        """
        os.environ["LOGLEVEL"] = "DEBUG"
        log = create_logger(__name__)
        self.assertEqual(
            log.name,
            __name__,
            msg="The logger should inherit the name of the file in which "
            f"`create_logger` is called ({__name__} instead of "
            f"{log.name})",
        )

        self.assertEqual(log.level, logging.DEBUG)
