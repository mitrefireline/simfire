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
        # Single Tile
        BL = (31, 112)
        TR = (32, 113)
        topographyGen = TopoLayer(BL, TR, '90m')
        topographyGen._get_dems()
        self.assertEqual(1, len(topographyGen.tif_filenames))

        # 2 Tiles
        BL = (31, 113)
        TR = (36, 111)
        topographyGen = TopoLayer(BL, TR, '90m')
        topographyGen._get_dems()
        self.assertEqual(2, len(topographyGen.tif_filenames))

        # 4 Tiles
        BL = (33.2, 112.4)
        TR = (37, 108)
        topographyGen = TopoLayer(BL, TR, '90m')
        topographyGen._get_dems()
        self.assertEqual(4, len(topographyGen.tif_filenames))

    def test__get_nearest_tile(self) -> None:
        '''
        Test that the call to _get_nearest_tile() runs properly.
        This method will calculate the closest 5 degrees (N, W) that
            contain the latitude and longitude specified
        '''
        # 2 Tiles
        BL = (31, 113)
        TR = (36, 111)
        topographyGen = TopoLayer(BL, TR, '90m')
        test_output_dems = ((30, 115), (35, 115))
        topographyGen._get_nearest_tile()
        self.assertEqual(topographyGen.five_deg_north_min, test_output_dems[0][0])
        self.assertEqual(topographyGen.five_deg_north_max, test_output_dems[1][0])
        self.assertEqual(topographyGen.five_deg_west_min, test_output_dems[0][1])
        self.assertEqual(topographyGen.five_deg_west_max, test_output_dems[1][1])

        # 3 Tiles
        BL = (32.4, 114.2)
        TR = (41, 112)
        topographyGen = TopoLayer(BL, TR, '90m')
        test_output_dems = ((30, 115), (35, 115), (40, 115))
        topographyGen._get_nearest_tile()
        self.assertEqual(topographyGen.five_deg_north_min, test_output_dems[0][0])
        self.assertEqual(topographyGen.five_deg_north_max, test_output_dems[2][0])
        self.assertEqual(topographyGen.five_deg_west_min, test_output_dems[0][1])
        self.assertEqual(topographyGen.five_deg_west_max, test_output_dems[1][1])

        # 4 Tiles
        BL = (33.2, 112.4)
        TR = (37, 108)
        topographyGen = TopoLayer(BL, TR, '90m')
        test_output_dems = ((30, 115), (30, 110), (35, 110), (35, 115))
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
        BL = (31, 113)
        TR = (36, 111)
        topographyGen = TopoLayer(BL, TR, '90m')
        test_output_dems = {'north': ((30, 115), (35, 115))}
        topographyGen._get_nearest_tile()
        test_dict = topographyGen._stack_tiles()
        self.assertEqual(test_output_dems, test_dict)

        # 4 Tiles
        BL = (33.2, 112.4)
        TR = (37, 108)
        topographyGen = TopoLayer(BL, TR, '90m')
        test_output_dems = {'square': ((30, 115), (30, 110), (35, 110), (35, 115))}
        topographyGen._get_nearest_tile()
        test_dict = topographyGen._stack_tiles()

        self.assertEqual(test_output_dems, test_dict)

        # 2 Tiles
        BL = (34.8, 117.2)
        TR = (31, 111)
        topographyGen = TopoLayer(BL, TR, '90m')
        test_output_dems = {'east': ((30, 120), (30, 115))}
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
        BL = (30.8, 116.5)
        TR = (33, 111.76)
        topographyGen = TopoLayer(BL, TR, '90m')
        data_layer = topographyGen._generate_contours()
        self.assertEqual(data_layer.shape, (2640, 5686, 1))

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

        # 2 Tiles easternly
        corners = [(30, 120), (30, 110), (35, 110), (35, 120)]
        BL = (31.5, 115.6)
        TR = (32.0, 114.8)
        topographyGen = TopoLayer(BL, TR, '90m')
        topographyGen._generate_lat_long(corners, 4, 4)
        # 8 longitudinal indices
        # 4 latitudinal indices
        self.assertTrue(topographyGen.elev_array.shape == (8, 4, 2))

        # 2 Tiles northernly
        corners = [(31.5, 120), (31.5, 118.2), (37, 118.2), (37, 120)]
        BL = (31.5, 120)
        TR = (37, 118.2)
        topographyGen = TopoLayer(BL, TR, '90m')
        topographyGen._generate_lat_long(corners, 4, 4)
        # 8 longitudinal indices
        # 4 latitudinal indices
        self.assertTrue(topographyGen.elev_array.shape == (4, 8, 2))

    def test__get_lat_long_bbox(self) -> None:
        '''
        Test that the call to _get_lat_long_bbox() runs properly.
        This method will update the corners of the array of DEM tiles loaded

        '''
        # 2 Tiles northernly
        BL = (31, 113)
        TR = (36, 111)
        corners = [(30, 115), (30, 110), (35, 110), (35, 115)]
        new_corner = (35, 115)
        output = [(30, 115), (30, 110), (40, 115), (40, 110)]
        topographyGen = TopoLayer(BL, TR, '90m')
        test_output = topographyGen._get_lat_long_bbox(corners, new_corner, stack='north')

        self.assertEqual(output, test_output)

        # 3 Tiles easternly (call twice with updated corners)
        BL = (31, 120)
        TR = (32, 108)
        corners = [(30, 120), (30, 115), (35, 115), (35, 120)]
        new_corner = (30, 115)
        output = [(30, 120), (30, 110), (35, 110), (35, 120)]
        topographyGen = TopoLayer(BL, TR, '90m')
        test_output = topographyGen._get_lat_long_bbox(corners, new_corner, stack='east')
        self.assertEqual(output, test_output)
        new_corner = (30, 110)
        output = [(30, 120), (30, 105), (35, 105), (35, 120)]
        test_output = topographyGen._get_lat_long_bbox(output, new_corner, stack='east')
        self.assertEqual(output, test_output)

    def test_save_contour_map(self) -> None:
        '''
        Test that the call to _save_contour_map() runs propoerly.
        '''
        pass
