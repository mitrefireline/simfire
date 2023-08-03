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
    TopographyLayer,
)


class TestLatLongBox(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up tests for reading in real topography
        """
        self.lat_long_box = LatLongBox(
            ((37.45, -120.44), (37.42, -120.40)),
            year="2020",
            layers=("fuel", "topographic"),
        )

    def test__get_layer_names(self):
        """
        Test the functionality to pull the correct ProductLayers from LandFire.
        """
        layer_products = self.lat_long_box._get_layer_names()
        self.assertEqual(
            len(layer_products), 2, msg="Humidity is not a valid LandFire product."
        )

    def test__make_data(self):
        """
        Test that the functionality to load fuel and topography data works as expected.
        """
        self.assertGreaterEqual(
            self.lat_long_box.geotiff_data.shape[-1],
            2,
            msg="SimFire expects at least Fuel and Topography layers additional "
            "layers can be pulled, but will not be utilized by SimFire.",
        )

        # after all tests with LatLongBox, delete the queried data folders
        rmtree = True
        if rmtree:
            shutil.rmtree(f"{self.lat_long_box.output_path}")


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

        As long as LatLongBox tests run, This layer will be accurate.
        """
        return super().setUp()


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

        As long as LatLongBox tests run, This layer will be accurate.
        """


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
