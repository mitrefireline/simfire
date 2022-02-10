import unittest
from ..topography import TopographyGen


class TestTopographyGen(unittest.TestCase):
    def setUp(self) -> None:
        '''
        Set up tests for reading in real topography
        '''

    def test_get_nearest_tile(self) -> None:

        latitude = (32.99, 37.3)
        longitude = (-116.7, -118.6)
        self.topographyGen = TopographyGen(latitude, longitude)

        test_output_dems = ((30, 120), (35, 120))
        output_dems = self.topographyGen._get_nearest_tile()

        self.assertEqual(test_output_dems, output_dems)

        latitude = (29.99, 34.99)
        longitude = (120.01, 115.01)
        test_output_dems = ((30, 120))
        output_dems = self.topographyGen = TopographyGen(latitude, longitude)
        self.assertEqual(test_output_dems, output_dems)
