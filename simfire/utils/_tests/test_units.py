import unittest

from ..units import (
    chains_to_feet_dozerline,
    chains_to_feet_handline,
    ftpm_to_mph,
    mph_to_ftpm,
    str_to_minutes,
)


class TestUnits(unittest.TestCase):
    def test_mph_to_ftpm(self) -> None:
        """
        Test to make sure the conversion from MPH to ft/min is correct
        """
        self.assertEqual(mph_to_ftpm(50), 50 * 88)

    def test_ftpm_to_mph(self) -> None:
        """
        Test to make sure the conversion from MPH to ft/min is correct
        """
        self.assertEqual(ftpm_to_mph(50), 50 / 88)

    def test_str_to_minutes(self) -> None:
        """
        Test to make sure the conversion from a string ('1d') turns into minutes ('1440')
        """
        self.assertEqual(str_to_minutes("24h"), 1440)
        self.assertEqual(str_to_minutes("1d"), 1440)
        self.assertEqual(str_to_minutes("1d 23h 60m"), 2880)

    def test_chains_to_feet_handline(self) -> None:
        """
        Test to make sure conversion from chains to feet calculation returns ints
        """
        self.assertEqual(chains_to_feet_handline(2.0), (132, 3))
        self.assertEqual(chains_to_feet_handline(0.5), (33, 3))

    def test_chains_to_feet_dozerline(self) -> None:
        """
        Test to make sure conversion from chains to feet calculation returns ints
        """
        self.assertEqual(chains_to_feet_dozerline(2.0), (132, 30))
        self.assertEqual(chains_to_feet_dozerline(0.5), (33, 30))
