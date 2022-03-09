import unittest
from ..topography import TopoLayer


class TestTopoLayer(unittest.TestCase):
    def setUp(self) -> None:
        '''
        Set up tests for reading in real topography
        '''

    def test__get_dems(self) -> None:
        '''
        Test that the call to _get_dems() runs properly.
        This method will generate a list of the DEMs in the fireline /nfs/
        '''

        resolution = 30

        # Single Tile
        center = (32.2, 115.6)
        area = 1600**2
        topographyGen = TopoLayer(center, area, resolution)
        topographyGen._get_dems()
        self.assertEqual(1, len(topographyGen.tif_filenames))

        # 2 Tiles
        center = (32.4, 115.0)
        area = 3200**2
        topographyGen = TopoLayer(center, area, resolution)
        topographyGen._get_dems()
        self.assertEqual(2, len(topographyGen.tif_filenames))

        # 4 Tiles
        center = (34.001, 116.008)
        area = 3200**2
        topographyGen = TopoLayer(center, area, resolution)
        topographyGen._get_dems()
        self.assertEqual(4, len(topographyGen.tif_filenames))

    def test__get_nearest_tile(self) -> None:
        '''
        Test that the call to _get_nearest_tile() runs properly.
        This method will calculate the closest 5 degrees (N, W) that
            contain the latitude and longitude specified
        '''
        # 2 Tiles
        center = (32.4, 115.004)
        area = 3200**2
        topographyGen = TopoLayer(center, area, 30)
        test_output_dems = ((32, 116), (32, 115))
        topographyGen._get_nearest_tile()
        self.assertEqual(topographyGen.five_deg_north_min, test_output_dems[0][0])
        self.assertEqual(topographyGen.five_deg_north_max, test_output_dems[1][0])
        self.assertEqual(topographyGen.five_deg_west_min, test_output_dems[1][1])
        self.assertEqual(topographyGen.five_deg_west_max, test_output_dems[0][1])

        # 4 Tiles
        center = (33.001, 115.001)
        area = 1800**2
        topographyGen = TopoLayer(center, area, 30)
        test_output_dems = ((32, 116), (32, 115), (33, 115), (34, 116))
        topographyGen._get_nearest_tile()
        self.assertEqual(topographyGen.five_deg_north_min, test_output_dems[0][0])
        self.assertEqual(topographyGen.five_deg_north_max, test_output_dems[2][0])
        self.assertEqual(topographyGen.five_deg_west_min, test_output_dems[1][1])
        self.assertEqual(topographyGen.five_deg_west_max, test_output_dems[0][1])

    def test__stack_tiles(self) -> None:
        '''
        Test that the call to _stack_tiles() runs properly.
        This method correctly stitches together tiles on (easternly, southernly, square)
        '''
        # 2 Tiles
        center = (32.001, 115.6)
        area = 1600**2
        resolution = 30
        topographyGen = TopoLayer(center, area, resolution)
        test_output_dems = {'north': ((31, 116), (32, 116))}
        topographyGen._get_nearest_tile()
        test_dict = topographyGen._stack_tiles()
        self.assertEqual(test_output_dems, test_dict)

        # 4 Tiles
        center = (34.9, 110.01)
        area = 108230**2
        topographyGen = TopoLayer(center, area, 90)
        test_output_dems = {'square': ((30, 115), (30, 110), (35, 110), (35, 115))}
        topographyGen._get_nearest_tile()
        test_dict = topographyGen._stack_tiles()

        self.assertEqual(test_output_dems, test_dict)

    def test__generate_contours(self) -> None:
        '''
        Test that the call to _generate_contours() runs propoerly.
        This method returns the data array containing the elevations within the
            specified bounding box region of the given latitudes and longitudes.
        '''
        # 2 Tiles (easternly)
        center = (33.4, 115.04)
        area = 3200**2
        resolution = 30
        topographyGen = TopoLayer(center, area, resolution)
        data_layer = topographyGen._generate_contours()
        self.assertEqual(data_layer.shape[0], data_layer.shape[1])

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

        # # 2 Tiles easternly
        corners = [(32, 116), (32, 114), (33, 114), (33, 114)]
        center = (32.4, 115.01)
        area = 3200**2
        resolution = 30
        topographyGen = TopoLayer(center, area, resolution)
        topographyGen._generate_lat_long(corners)
        self.assertEqual(topographyGen.elev_array.shape, (7224, 3612, 2))

        # 2 Tiles northernly
        corners = [(31.5, 119), (31.5, 118), (34, 118), (34, 119)]
        center = (32.001, 115.6)
        area = 3200**2
        resolution = 30
        topographyGen = TopoLayer(center, area, resolution)
        topographyGen._generate_lat_long(corners)
        self.assertEqual(topographyGen.elev_array.shape, (3612, 7224, 2))

    def test__get_lat_long_bbox(self) -> None:
        '''
        Test that the call to _get_lat_long_bbox() runs properly.
        This method will update the corners of the array of DEM tiles loaded

        '''
        # 2 Tiles northernly
        center = (32.001, 115.6)
        area = 3200**2
        resolution = 30
        corners = [(31, 116), (31, 115), (32, 115), (32, 116)]
        new_corner = (32, 116)
        output = [(31, 116), (31, 115), (33, 115), (33, 116)]
        topographyGen = TopoLayer(center, area, resolution)
        test_output = topographyGen._get_lat_long_bbox(corners, new_corner, stack='north')

        self.assertEqual(output, test_output)

    def test_save_contour_map(self) -> None:
        '''
        Test that the call to _save_contour_map() runs propoerly.
        '''
        pass
