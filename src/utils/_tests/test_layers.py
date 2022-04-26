import unittest
from ..layers import FunctionalElevationLayer, LatLongBox, TopographyLayer
from ..layers import FunctionalFuelLayer, OperationalFuelLayer
from ...world.parameters import Fuel


class TestLatLongBox(unittest.TestCase):
    def setUp(self) -> None:
        '''
        Set up tests for reading in real topography
        '''

    def test__get_nearest_tile(self) -> None:
        '''
        Test that the call to _get_nearest_tile() runs properly.
        This method will calculate the closest 5 degrees (N, W) that
            contain the latitude and longitude specified
        '''
        resolution = 30

        # 2 Tiles
        center = (33.4, 116.004)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        test_output_dems = ((33, 117), (33, 115))
        self.assertEqual(lat_long_box.five_deg_north_min, test_output_dems[0][0])
        self.assertEqual(lat_long_box.five_deg_north_max, test_output_dems[1][0])
        self.assertEqual(lat_long_box.five_deg_west_min, test_output_dems[1][1])
        self.assertEqual(lat_long_box.five_deg_west_max, test_output_dems[0][1])

        # 4 Tiles
        center = (35.001, 117.001)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        test_output_dems = ((34, 118), (34, 117), (35, 117), (35, 118))
        self.assertEqual(lat_long_box.five_deg_north_min, test_output_dems[0][0])
        self.assertEqual(lat_long_box.five_deg_north_max, test_output_dems[2][0])
        self.assertEqual(lat_long_box.five_deg_west_min, test_output_dems[1][1])
        self.assertEqual(lat_long_box.five_deg_west_max, test_output_dems[0][1])

    def test__stack_tiles(self) -> None:
        '''
        Test that the call to _stack_tiles() runs properly.
        This method correctly stitches together tiles on (easternly, southernly, square)
        '''

        # 2 Tiles
        resolution = 30
        center = (35.001, 115.6)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        test_output_dems = {'north': ((34, 116), (35, 116))}
        self.assertEqual(test_output_dems, lat_long_box.tiles)

        # 4 Tiles
        resolution = 90
        center = (34.99, 115.001)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        test_output_dems = {'square': ((30, 120), (30, 115), (40, 115), (40, 120))}
        self.assertEqual(test_output_dems, lat_long_box.tiles)

    def test__generate_lat_long(self) -> None:
        '''
        Test that the call to _genrate_lat_long() runs properly.
        This method first creates an array of the latitude and longitude coords of all
            DEMs contained withing the specified lat/long region when MERITLayer
            is initialized. It will correctly calculate the corners of the array
            and fill the array with lat/long pairs using 5degs/6000 pixels
            or ~3 arc-seconds at every point.
        It will then find the correct bounds of the specified lat/long region to
            pull elevation data.
        '''
        resolution = 30

        # # 2 Tiles easternly
        center = (36.4, 118.01)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        self.assertEqual(lat_long_box.elev_array.shape, (7224, 3612, 2))

        # 2 Tiles northernly
        center = (34.001, 115.6)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        self.assertEqual(lat_long_box.elev_array.shape, (3612, 7224, 2))

    def test__get_lat_long_bbox(self) -> None:
        '''
        Test that the call to _get_lat_long_bbox() runs properly.
        This method will update the corners of the array of DEM tiles loaded

        '''
        resolution = 30

        # 2 Tiles northernly
        center = (34.001, 115.6)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        output = [(33, 116), (33, 115), (35, 115), (35, 116)]
        self.assertEqual(output, lat_long_box.corners)

    def test_save_contour_map(self) -> None:
        '''
        Test that the call to _save_contour_map() runs propoerly.
        '''
        resolution = 30
        # Single Tile
        center = (33.5, 116.8)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        topo_layer = TopographyLayer(lat_long_box)
        lat_long_box._save_contour_map(topo_layer.data)


class TestOperationalTopographyLayer(unittest.TestCase):
    def setUp(self) -> None:
        '''

        '''

    def test__make_contour_and_data(self) -> None:
        '''
        Test that the call to _generate_contours() runs propoerly.
        This method returns the data array containing the elevations within the
            specified bounding box region of the given latitudes and longitudes.

        NOTE: This method should always return a square
        '''
        resolution = 30
        # 2 Tiles (easternly)
        center = (33.4, 115.04)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        topographyGen = TopographyLayer(lat_long_box)
        self.assertEqual(topographyGen.data.shape[0], topographyGen.data.shape[1])

    def test__get_dems(self) -> None:
        '''
        Test that the call to _get_dems() runs properly.
        This method will generate a list of the DEMs in the fireline /nfs/
        '''

        resolution = 30
        # Single Tile
        center = (35.2, 115.6)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        topographyGen = TopographyLayer(lat_long_box)
        self.assertEqual(1, len(topographyGen.tif_filenames))

        # 2 Tiles
        center = (38.4, 115.0)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        topographyGen = TopographyLayer(lat_long_box)
        self.assertEqual(2, len(topographyGen.tif_filenames))

        # 4 Tiles
        center = (34.001, 116.008)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        topographyGen = TopographyLayer(lat_long_box)
        self.assertEqual(4, len(topographyGen.tif_filenames))


class TestFunctionalElevationLayer(unittest.TestCase):
    def setUp(self) -> None:
        # Create arbitrary function to test
        self.fn = lambda x, y: x + y
        # Set arbitrary screen size
        self.screen_size = (32, 32)
        return super().setUp()

    def test_data(self) -> None:
        '''
        Test that the FuncitonalElevationLayer creates the correct data
        '''
        height = self.screen_size[0]
        width = self.screen_size[1]
        layer = FunctionalElevationLayer(height, width, self.fn)
        correct_data_shape = self.screen_size + (1, )
        self.assertTupleEqual(correct_data_shape,
                              layer.data.shape,
                              msg=f'The layer data has shape {layer.data.shape}, '
                              f'but should have shape {correct_data_shape}')


class TestOperationalFuelLayer(unittest.TestCase):
    def setUp(self) -> None:
        '''

        '''

    def test_make_data(self) -> None:
        '''

        '''
        resolution = 30
        # 2 Tiles (easternly)
        center = (33.4, 115.04)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        FuelGen = OperationalFuelLayer(lat_long_box)
        self.assertEqual(FuelGen.data.shape[0], FuelGen.data.shape[1])

    def test_get_fuel_dems(self) -> None:
        '''

        '''

        resolution = 30
        # Single Tile
        center = (35.2, 115.6)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        FuelGen = OperationalFuelLayer(lat_long_box)
        self.assertEqual(1, len(FuelGen.tif_filenames))

        # 2 Tiles
        center = (38.4, 115.0)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        FuelGen = OperationalFuelLayer(lat_long_box)
        self.assertEqual(2, len(FuelGen.tif_filenames))

        # 4 Tiles
        center = (34.001, 116.008)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        FuelGen = OperationalFuelLayer(lat_long_box)
        self.assertEqual(4, len(FuelGen.tif_filenames))

    def test_make_image(self) -> None:
        '''

        '''
        pass


class TestFunctionalFuelLayer(unittest.TestCase):
    def setUp(self) -> None:
        '''

        '''
        self.height = 1000
        self.width = 1000
        self.fuel_fn = Fuel
        self.FunctionalFuel = FunctionalFuelLayer(self.height, self.width, self.fuel_fn)

    def test_make_data(self) -> None:
        '''

        '''

    def test_make_image(self) -> None:
        '''

        '''

        self.assertCountEqual(self.FunctionalFuel.image.shape,
                              self.screen_size,
                              msg=('The terrain fuels have shape '
                                   f'{self.terrain.fuels.shape}, but should have '
                                   f'shape {self.screen_size}'))

    def test_load_texture(self) -> None:
        '''

        '''

    def test_update_texture_dryness(self) -> None:
        '''

        '''
