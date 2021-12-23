import unittest

from ..units import mph_to_ftpm, ftpm_to_mph


class TestUnits(unittest.TestCase):
    def test_mph_to_ftpm(self) -> None:
        '''
        Test to make sure the conversion from MPH to ft/min is correct
        '''
        self.assertEqual(mph_to_ftpm(50), 50 * 88)

    def test_ftpm_to_mph(self) -> None:
        '''
        Test to make sure the conversion from MPH to ft/min is correct
        '''
        self.assertEqual(ftpm_to_mph(50), 50 / 88)
