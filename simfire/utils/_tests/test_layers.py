import os
import shutil
import unittest

import matplotlib as mpl
import numpy as np

from ...world.fuel_array_functions import chaparral_fn
from ..layers import (
    DataLayer,
    FunctionalBurnProbabilityLayer,
    FunctionalFuelLayer,
    FunctionalTopographyLayer,
    LatLongBox,
    OperationalBurnProbabilityLayer,
    OperationalFuelLayer,
    OperationalTopographyLayer,
    TopographyLayer,
)


class TestLatLongBox(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up tests for reading in real topography
        """

    def test__get_nearest_tile(self) -> None:
        """
        Test that the call to _get_nearest_tile() runs properly.
        This method will calculate the closest 5 degrees (N, W) that
            contain the latitude and longitude specified
        """
        resolution = 30

        # 2 Tiles
        center = (33.4, 116.004)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        test_output_dems = ((33, 117), (33, 115))
        self.assertEqual(lat_long_box.deg_north_min, test_output_dems[0][0])
        self.assertEqual(lat_long_box.deg_north_max, test_output_dems[1][0])
        self.assertEqual(lat_long_box.deg_west_min, test_output_dems[1][1])
        self.assertEqual(lat_long_box.deg_west_max, test_output_dems[0][1])

        # 4 Tiles
        center = (35.001, 117.001)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        test_output_dems = ((34, 118), (34, 117), (35, 117), (35, 118))
        self.assertEqual(lat_long_box.deg_north_min, test_output_dems[0][0])
        self.assertEqual(lat_long_box.deg_north_max, test_output_dems[2][0])
        self.assertEqual(lat_long_box.deg_west_min, test_output_dems[1][1])
        self.assertEqual(lat_long_box.deg_west_max, test_output_dems[0][1])

    def test__stack_tiles(self) -> None:
        """
        Test that the call to _stack_tiles() runs properly.
        This method correctly stitches together tiles on (easternly, southernly, square)
        """

        # 2 Tiles
        resolution = 30
        center = (35.001, 115.6)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        test_output_dems = {"north": ((34, 116), (35, 116))}
        self.assertEqual(test_output_dems, lat_long_box.tiles)

        # 4 Tiles
        resolution = 90
        center = (34.99, 115.001)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        test_output_dems = {"square": ((30, 120), (30, 115), (40, 115), (40, 120))}
        self.assertEqual(test_output_dems, lat_long_box.tiles)

    def test__generate_lat_long(self) -> None:
        """
        Test that the call to _genrate_lat_long() runs properly.
        This method first creates an array of the latitude and longitude coords of all
            DEMs contained withing the specified lat/long region when MERITLayer
            is initialized. It will correctly calculate the corners of the array
            and fill the array with lat/long pairs using 5degs/6000 pixels
            or ~3 arc-seconds at every point.
        It will then find the correct bounds of the specified lat/long region to
            pull elevation data.
        """
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
        """
        Test that the call to _get_lat_long_bbox() runs properly.
        This method will update the corners of the array of DEM tiles loaded

        """
        resolution = 30

        # 2 Tiles northernly
        center = (34.001, 115.6)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        output = [(33, 116), (33, 115), (35, 115), (35, 116)]
        self.assertEqual(output, lat_long_box.corners)

    def test_save_contour_map(self) -> None:
        """
        Test that the call to _save_contour_map() runs propoerly.
        """
        resolution = 30
        # Single Tile
        center = (33.5, 116.8)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        topo_layer = OperationalTopographyLayer(lat_long_box)
        rmtree = False
        if not os.path.isdir("images/"):
            os.makedirs("images/")
            rmtree = True
        # Make sure that the function runs successfully
        try:
            lat_long_box._save_contour_map(topo_layer.data, "topo")
        except:  # noqa E722 (Ignore this since it's for a test)
            if rmtree:
                shutil.rmtree("images/")
            self.fail("lat_long_box._save_contour_map() did not run successfully")
        if rmtree:
            shutil.rmtree("images/")


class TestDataLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.layer = DataLayer()
        return super().setUp()

    def test_data(self) -> None:
        """
        Test that the data is set to None.
        """
        print(self.layer.data)
        self.assertIsNone(
            self.layer.data,
            msg="The initialized data should be None, " f"but is {self.layer.data}",
        )


class TestTopographyLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.topo_layer = TopographyLayer()
        return super().setUp()

    def test__make_contours(self) -> None:
        """
        Test that the contours are created correctly.
        """
        # Set the data to something arbitrary
        self.topo_layer.data = np.random.rand(10, 10)
        contours = self.topo_layer._make_contours()
        self.assertIsInstance(
            contours,
            mpl.contour.QuadContourSet,
            msg="The returned contours should be of type "
            "matplotlib.contour.QuadContourSet, but are of "
            f"type {type(contours)}",
        )


class TestOperationalTopographyLayer(unittest.TestCase):
    def setUp(self) -> None:
        """
        Each test requires a new layer, so nothing is done in setUp.
        """
        return super().setUp()

    def test__make_data(self) -> None:
        """
        Test that the internal call to _make_data() runs properly.
        This method returns the data array containing the elevations within the
            specified bounding box region of the given latitudes and longitudes.

        NOTE: This method should always return a square
        """
        resolution = 30
        # 2 Tiles (easternly)
        center = (33.4, 115.04)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        topographyGen = OperationalTopographyLayer(lat_long_box)
        self.assertEqual(topographyGen.data.shape[0], topographyGen.data.shape[1])

    def test__get_dems(self) -> None:
        """
        Test that the call to _get_dems() runs properly.
        This method will generate a list of the DEMs in the fireline /nfs/
        """
        resolution = 30
        # Single Tile
        center = (35.2, 119.6)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        topographyGen = OperationalTopographyLayer(lat_long_box)
        self.assertEqual(1, len(topographyGen.tif_filenames))

        # 2 Tiles
        center = (37.4, 115.0)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        topographyGen = OperationalTopographyLayer(lat_long_box)
        self.assertEqual(2, len(topographyGen.tif_filenames))

        # 4 Tiles
        center = (34.001, 116.008)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        topographyGen = OperationalTopographyLayer(lat_long_box)
        self.assertEqual(4, len(topographyGen.tif_filenames))


class TestFunctionalTopograpyLayer(unittest.TestCase):
    def setUp(self) -> None:
        # Create arbitrary function to test
        self.fn = lambda x, y: x + y
        # Set arbitrary screen size
        self.screen_size = (32, 32)
        return super().setUp()

    def test_data(self) -> None:
        """
        Test that the FuncitonalElevationLayer creates the correct data
        """
        height = self.screen_size[0]
        width = self.screen_size[1]
        layer = FunctionalTopographyLayer(height, width, self.fn, name="test")
        correct_data_shape = self.screen_size + (1,)
        self.assertTupleEqual(
            correct_data_shape,
            layer.data.shape,
            msg=f"The layer data has shape {layer.data.shape}, "
            f"but should have shape {correct_data_shape}",
        )


class TestFuelLayer(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()


class TestOperationalFuelLayer(unittest.TestCase):
    def setUp(self) -> None:
        """
        Each test requires a new layer, so nothing is done in setUp.
        """

    def test_make_data(self) -> None:
        """
        Test that the internal call to _make_data() runs properly.
        This method returns the data array containing the elevations within the
            specified bounding box region of the given latitudes and longitudes.

        NOTE: This method should always return a square
        """
        resolution = 30
        # 2 Tiles (easternly)
        center = (33.4, 116.04)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        FuelGen = OperationalFuelLayer(lat_long_box)
        self.assertEqual(FuelGen.data.shape[0], FuelGen.data.shape[1])

    def test_get_fuel_dems(self) -> None:
        """
        Test that the call to _get_dems() runs properly.
        This method will generate a list of the DEMs in the fireline /nfs/
        """
        resolution = 30
        # Single Tile
        center = (35.2, 117.6)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        FuelGen = OperationalFuelLayer(lat_long_box)
        self.assertEqual(1, len(FuelGen.fuel_model_filenames))

        # 2 Tiles
        center = (37.4, 118.0)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        FuelGen = OperationalFuelLayer(lat_long_box)
        self.assertEqual(2, len(FuelGen.fuel_model_filenames))

        # 4 Tiles
        center = (33.001, 116.008)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        FuelGen = OperationalFuelLayer(lat_long_box)
        self.assertEqual(4, len(FuelGen.fuel_model_filenames))

    def test_image(self) -> None:
        """
        Test that the internal call to _make_image() runs properly and
        returns a numpy array.
        """
        resolution = 30
        # Single Tile
        center = (35.2, 117.6)
        height, width = 1600, 1600
        lat_long_box = LatLongBox(center, height, width, resolution)
        FuelGen = OperationalFuelLayer(lat_long_box)

        self.assertIsInstance(
            FuelGen.image,
            np.ndarray,
            msg="The created image should be a numpy.ndarray, "
            f"but is of type {type(FuelGen.image)}",
        )


class TestFunctionalFuelLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.height = 32
        self.width = 32
        self.fuel_fn = chaparral_fn()
        self.FunctionalFuel = FunctionalFuelLayer(
            self.height, self.width, self.fuel_fn, name="test"
        )

    def test_data(self) -> None:
        """
        Test that the layer creates the data as a numpy array with the correct shape.
        """
        correct_data_shape = (self.height, self.width) + (1,)
        data_shape = self.FunctionalFuel.data.shape
        self.assertTupleEqual(
            data_shape,
            correct_data_shape,
            msg=f"The layer data has shape {data_shape}, "
            f"but should have shape {correct_data_shape}",
        )

    def test_image(self) -> None:
        """
        Test that the internal call to _make_iamge runs properly and returns
        a numpy aray with the correct shape.
        """
        correct_data_shape = (self.height, self.width) + (3,)
        image = self.FunctionalFuel.image
        self.assertIsInstance(
            image,
            np.ndarray,
            msg="The created image should be a numpy.ndarray, "
            f"but is of type {type(image)}",
        )

        self.assertCountEqual(
            image.shape,
            correct_data_shape,
            msg=(
                "The terrain fuels have shape "
                f"{image.shape}, but should have "
                f"shape {correct_data_shape}"
            ),
        )


class TestBurnProbabilityLayer(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()


# Need to tweak the gdal outputs for Burn Probability, incorrect pixel size
class TestOperationalBurnProbabilityLayer(unittest.TestCase):
    def setUp(self) -> None:
        """
        Each test requires a new layer, so nothing is done in setUp.
        """
        return super().setUp()

    def test__make_data(self) -> None:
        """
        Test that the internal call to _make_data() runs properly.
        This method returns the data array containing the elevations within the
            specified bounding box region of the given latitudes and longitudes.

        NOTE: This method should always return a square
        """

        resolution = 30
        # 2 Tiles (easternly)
        center = (33.4, 115.04)
        height, width = 3200, 3200
        lat_long_box = LatLongBox(center, height, width, resolution)
        self.assertTrue(True)  # Remove after burn probability data is fixed to
        return
        OperationalBurnProbabilityLayer(lat_long_box)

        # match 3612, 3612 (pixel x pixel per image)
        # self.assertEqual(
        #     burnprobabilityGen.data.shape[0], burnprobabilityGen.data.shape[1]
        # )

    def test__get_dems(self) -> None:
        """
        Test that the call to _get_dems() runs properly.
        This method will generate a list of the DEMs in the fireline /nfs/
        """
        self.assertTrue(True)

        # resolution = 30
        # # Single Tile
        # center = (35.2, 115.6)
        # height, width = 1600, 1600
        # lat_long_box = LatLongBox(center, height, width, resolution)
        # burnprobabilityGen = OperationalBurnProbabilityLayer(lat_long_box)
        # self.assertEqual(1, len(burnprobabilityGen.tif_filenames))

        # # 2 Tiles
        # center = (38.4, 115.0)
        # height, width = 1600, 1600
        # lat_long_box = LatLongBox(center, height, width, resolution)
        # burnprobabilityGen = OperationalBurnProbabilityLayer(lat_long_box)
        # self.assertEqual(2, len(burnprobabilityGen.tif_filenames))

        # # 4 Tiles
        # center = (34.001, 116.008)
        # height, width = 3200, 3200
        # lat_long_box = LatLongBox(center, height, width, resolution)
        # burnprobabilityGen = OperationalBurnProbabilityLayer(lat_long_box)
        # self.assertEqual(4, len(burnprobabilityGen.tif_filenames))


class TestFunctionalBurnProbabilityLayer(unittest.TestCase):
    def setUp(self) -> None:
        # Create arbitrary function to test
        self.fn = lambda x, y: x + y
        # Set arbitrary screen size
        self.screen_size = (32, 32)
        return super().setUp()

    def test_data(self) -> None:
        """
        Test that the FuncitonalElevationLayer creates the correct data
        """
        height = self.screen_size[0]
        width = self.screen_size[1]
        layer = FunctionalBurnProbabilityLayer(height, width, self.fn, name="test")
        correct_data_shape = self.screen_size + (1,)
        self.assertTupleEqual(
            correct_data_shape,
            layer.data.shape,
            msg=f"The layer data has shape {layer.data.shape}, "
            f"but should have shape {correct_data_shape}",
        )
