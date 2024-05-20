import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import landfire
import matplotlib.pyplot as plt
import numpy as np
from geotiff import GeoTiff
from landfire.product.enums import ProductRegion, ProductTheme, ProductVersion
from landfire.product.search import ProductSearch
from matplotlib.contour import QuadContourSet
from PIL import Image

from ..enums import (
    DRY_TERRAIN_BROWN_IMG,
    TERRAIN_TEXTURE_PATH,
    FuelModelRGB13,
    FuelModelToFuel,
)
from ..utils.log import create_logger
from ..world.elevation_functions import ElevationFn
from ..world.fuel_array_functions import FuelArrayFn
from ..world.parameters import Fuel

log = create_logger(__name__)


class LandFireLatLongBox:
    """
    Class that gets all relevant operational data.

    Currently only support using Fuel and Elevation data within SimFire.
    """

    def __init__(
        self,
        points: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (37.45, -120.44),
            (37.35, -120.22),
        ),
        year: str = "2020",
        layers: Tuple[str, str] = ("fuel", "topographic"),
        height: int = 4500,
        width: int = 4500,
    ) -> None:
        """
        This class of methods will get initialized with the config using the lat/long
        bounding box.

        Real-world is measured in meters
        Data is measured in pixels corresponding to the resolution
            i.e: resolution = 30m = 1 pixel

        In geospatial world:
            30m == 0.00027777777803598015 degrees in lat/long space
            caveat: ESPG 4326, WGS84

        """

        self.points = points
        self.year = year
        self.layers = layers

        # FIX FOR CRS CHANGES
        # pad the original BR points by a marginal amount of area --> 90 meters each side
        # to ensure that the output shape matches the specified/desired output shape

        # Padding only one corner allows us to use the TL as a starting point to
        # crop from

        self.pad_distance = 0.00027777777803598015 * 100

        # sepcify the output paths of the Landfire python-client query
        self.product_layers_path = Path(
            f"lf_{abs(self.points[0][1])}_{self.points[0][0]}_"
            f"{abs(self.points[1][1]+self.pad_distance)}_"
            f"{self.points[1][0]-self.pad_distance}.zip"
        )

        if os.environ.get("SF_HOME") is None:
            sf_path = Path().home() / ".simfire"
        else:
            sf_path = Path(str(os.environ.get("SF_HOME")))
            if not sf_path.exists():
                log.info(f"Creating SF_HOME directory: {sf_path}")
                sf_path.mkdir(parents=True, exist_ok=True)

        self.output_path = (
            sf_path / f"landfire/{self.year}/{self.product_layers_path.stem}/"
        )

        log.info(f"Saving LandFire data to: {self.output_path}")

        # make each a layer a global variable that we "fill"
        self.fuel = np.array([])
        self.topography = np.array([])

        # check if we've already pulled this data before
        exists = self._check_paths()
        if exists:
            self._make_data()
        else:
            self.layer_products = self._get_layer_names()
            self.query_lat_lon_layer_data()
            self._make_data()

        # FIX FOR CRS CHANGES
        # crop data to new pixel_height and pixel_width
        pixel_height = int(np.floor(height / 30))  # round down to nearest int
        pixel_width = int(np.floor(width / 30))  # round down to nearest int
        self.fuel = self.fuel[:pixel_height, :pixel_width]
        self.topography = self.topography[:pixel_height, :pixel_width]

        log.info(
            f"Output shape of Fire Map: {height}m x {width}m "
            f"--> {self.fuel.shape} in pixel space"
        )

    def _check_paths(self):
        """
        Check to verify if this exact data has already been pulled.
            This is unlikely, but could save time if so.

        Assume we always pull at least Fuel and Topography data.

        TODO: Currently we do not have the functionality to find points within an
                existing LatLongBox that has already been downloaded.
                This could greatly improve functionality and speed up training interations
        """
        if self.output_path.exists():
            tifs = [str(t) for t in self.output_path.glob("*.tif")]
            if len(tifs) == 0:
                log.info(
                    f"The output path, {self.output_path}, exists, but does not "
                    "contain any tif files. Returning false to ensure that data "
                    "for this area is pulled."
                )
                return False
            else:
                log.debug(
                    "Data for this area already exists. Loading from file: "
                    f"{self.output_path}"
                )
                return True
        else:
            return False

    def _get_layer_names(self) -> Dict[str, str]:
        """
        Functionality to ge the LandFire Product names for the Layers specified.

        The way this is written is such that you could add other layers
            of interest that LandFire provides:
                - Slope (degrees)
                -  operational roads
                - fuel vegetation
                - fuel / canopy height
                etc.

        TODO: make the ProductSearch less clunky

        Elevation does not really change from year to year, so LandFire only provides
            one


        """
        # use a dictionary to keep track of the different/possible layers
        layer_products = {}
        for layer in self.layers:
            # check if layer is in the LandFire ProductThemes
            valid_layer = layer in list(ProductTheme)
            if valid_layer:
                if layer == "fuel":
                    code = "FBFM13"
                    product_theme = ProductTheme.fuel
                    if str(self.year) == "2019" or str(self.year) == "2020":
                        product_version = ProductVersion.lf_2016_remap
                    else:
                        product_version = ProductVersion.lf_2020
                elif layer == "topographic":
                    code = "ELEV"
                    layer = "topographic"  # because LandFire has weird naming conventions
                    product_version = ProductVersion.lf_2020
                    product_theme = ProductTheme.topographic

                search: ProductSearch = ProductSearch(
                    codes=[code],
                    versions=[product_version],
                    themes=[product_theme],
                    regions=[ProductRegion.US],
                )

                layer_search = search.get_layers()
                # 2019 and 2020 for fuel have the same codes/name/version in LandFire
                if layer == "fuel":
                    if str(self.year) == "2019":
                        layer_products[layer] = layer_search[0]
                    else:
                        layer_products[layer] = layer_search[1]
                else:
                    layer_products[layer] = layer_search[0]

            else:
                log.warning(
                    f"Currently only supports the following LandFire Products "
                    "within SimFire: "
                    f"{ProductTheme.fuel}, {ProductTheme.topographic}."
                )

        return layer_products

    def query_lat_lon_layer_data(self):
        """
        Functionality to get the indices and values of the layers at a queried point.

        For GIS data you need to calculate the offsets to get the array indice

        Arguments:
            layer: the gdal layer to query
            band: the gdal raster band that contains the data

        """
        self.output_path.mkdir(parents=True, exist_ok=True)

        lf = landfire.Landfire(
            bbox=f"{self.points[0][1]} {self.points[0][0]} \
                {self.points[1][1]+self.pad_distance} \
                {self.points[1][0]-self.pad_distance}",
            output_crs="4326",
        )
        # assume the order of data retrieval stays the same
        lf.request_data(
            layers=self.layer_products.values(),
            output_path=str(self.output_path / self.product_layers_path),
        )

        shutil.unpack_archive(
            self.output_path / self.product_layers_path, self.output_path
        )

    def _make_data(self):
        """
        Functionality to read in tif data for layers, and create iamge data.

        NOTE: for now we assume fuel and topography are always pulled in the respective
                order, but this could be used in the future:

        ```
            for i in range(len(self.layers)):
                globals()[self.layers[i]] = np.array(self.geotiff_data[:, :, i])
        ```

        """
        tifs = [str(t) for t in self.output_path.glob("*.tif")][0]

        # the order the data was requested is the order of the Band in the Tif file
        geo_tiff = GeoTiff(tifs, crs_code=4326)
        self.geotiff_data = geo_tiff.read()

        self.fuel = np.array(self.geotiff_data[:, :, 0])
        self.topography = np.array(self.geotiff_data[:, :, 1])


class LatLongBox:
    """
    Base class for original LatLongBox functionality.

    Only used for typing purposes or until LadFire provides
        BurnProbabilty data.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initializes variables/methods that the BurnProbabilty layer types.
        """
        self.resolution: str = "30m"
        self.tiles: dict = {}
        self.bl: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 0))
        self.tr: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 0))


class DataLayer:
    """
    Base class for any data that affects the terrain.
    The data in this class should have a value for every pixel in the terrain.
    """

    def __init__(self) -> None:
        """
        This parent class only exists to set a base value for self.data
        """
        self.data: Optional[np.ndarray] = None


class BurnProbabilityLayer(DataLayer):
    """
    Base class for use with operational and procedurally generated
    fuel data. This class implements the code needed to
    create the terrain image to use with the display.
    """

    def __init__(self) -> None:
        """
        Simple call to the parent DataLayer class.

        Arguments:
            None

        Returns:
            None
        """
        super().__init__()
        self.data: np.ndarray
        self.image: np.ndarray

    def _make_contours(self) -> QuadContourSet:
        """
        Use the data in self.data to compute the contour lines.

        Arguments:
            None

        Returns:
            contours: The matplotlib contour set used for plotting
        """
        contours = plt.contour(self.data.squeeze(), origin="upper")
        plt.close()
        return contours


class OperationalBurnProbabilityLayer(BurnProbabilityLayer):
    def __init__(self, lat_long_box: LatLongBox, path: Path) -> None:
        """
        Initialize the elevation layer by retrieving the correct topograpchic data
        and computing the area

        Arguments:
            center: The lat/long coordinates of the center point of the screen.
            height: The height of the screen size (meters).
            width: The width of the screen size (meters).
            resolution: The resolution to get data (meters).
        """
        super().__init__()
        self.lat_long_box = lat_long_box
        self.path = Path(path) / "risk"
        res = str(self.lat_long_box.resolution) + "m"
        self.datapath = self.path / res

        # TODO: Add check here if resolution isnt available

        self.data = self._make_data()
        self.contours = self._make_contours()

    def _make_data(self) -> np.ndarray:
        self._get_dems()
        data = Image.open(self.tif_filenames[0])
        data = np.array(data, dtype=np.float32)
        # flip axis because latitude goes up but numpy will read it down
        data = np.flip(data, 0)
        data = np.expand_dims(data, axis=-1)

        for key, _ in self.lat_long_box.tiles.items():
            if key == "single":
                # simple case
                tr = (self.lat_long_box.bl[0][0], self.lat_long_box.tr[1][0])
                bl = (self.lat_long_box.tr[0][0], self.lat_long_box.bl[1][0])
                return data[tr[0] : bl[0], tr[1] : bl[1]]
            tmp_array = data
            for idx, dem in enumerate(self.tif_filenames[1:]):
                tif_data = Image.open(dem)
                tif_data = np.array(tif_data, dtype=np.float32)
                # flip axis because latitude goes up but numpy will read it down
                tif_data = np.flip(tif_data, 0)
                tif_data = np.expand_dims(tif_data, axis=-1)

                if key == "north":
                    # stack tiles along axis = 0 -> leftmost: bottom, rightmost: top
                    data = np.concatenate((data, tif_data), axis=0)
                elif key == "east":
                    # stack tiles along axis = 2 -> leftmost, rightmost
                    data = np.concatenate((data, tif_data), axis=1)
                elif key == "square":
                    if idx + 1 == 1:
                        data = np.concatenate((data, tif_data), axis=1)
                    elif idx + 1 == 2:
                        tmp_array = tif_data
                    elif idx + 1 == 3:
                        tmp_array = np.concatenate((tif_data, tmp_array), axis=1)
                        data = np.concatenate((data, tmp_array), axis=0)

        tr = (self.lat_long_box.bl[0][0], self.lat_long_box.tr[1][0])
        bl = (self.lat_long_box.tr[0][0], self.lat_long_box.bl[1][0])
        data_array = data[tr[0] : bl[0], tr[1] : bl[1]]
        # Convert from meters to feet for use with Rothermel
        data_array = 3.28084 * data_array
        return data_array

    def _get_dems(self) -> None:
        """
        Uses the outputed tiles and sets `self.tif_filenames`
        """
        self.tif_filenames = []

        for _, ranges in self.lat_long_box.tiles.items():
            for range in ranges:
                (five_deg_n, five_deg_w) = range
                tif_data_region = Path(f"n{five_deg_n}w{five_deg_w}.tif")
                tif_file = self.datapath / tif_data_region
                self.tif_filenames.append(tif_file)


class FunctionalBurnProbabilityLayer(BurnProbabilityLayer):
    """
    Layer that stores elevation data computed from a function.
    """

    def __init__(self, height, width, elevation_fn: ElevationFn, name: str) -> None:
        """
        Initialize the elvation layer by computing the elevations and contours.

        Arguments:
            height: The height of the data layer
            width: The width of the data layer
            elevation_fn: A callable function that converts (x, y) coorindates to
                          elevations.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.name = name

        self.data = self._make_data(elevation_fn)
        self.contours = self._make_contours()

    def _make_data(self, elevation_fn: ElevationFn) -> np.ndarray:
        """
        Use self.elevation_fn to make the elevation data layer.

        Arguments:
            elevation_fn: The function that maps (x, y) points to elevations

        Returns:
            A numpy array containing the elevation data
        """
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)
        elevation_fn_vect = np.vectorize(elevation_fn)
        elevations = elevation_fn_vect(X, Y)
        # Expand third dimension to align with data layers
        elevations = np.expand_dims(elevations, axis=-1)

        return elevations


class TopographyLayer(DataLayer):
    """
    Base class for use with operational and procedurally generated
    topographic/elevation data. This class implements the code needed to
    create the contour image to use with the display.
    """

    def __init__(self) -> None:
        """
        Simple call to the parent DataLayer class.

        Arguments:
            None

        Returns:
            None
        """
        super().__init__()
        self.data: np.ndarray
        self.image: np.ndarray

    def _make_contours(self) -> QuadContourSet:
        """
        Use the data in self.data to compute the contour lines.

        Arguments:
            None

        Returns:
            contours: The matplotlib contour set used for plotting
        """
        contours = plt.contour(self.data.squeeze(), origin="upper")
        plt.close()
        return contours


class OperationalTopographyLayer(TopographyLayer):
    def __init__(self, LandFireLatLongBox: LandFireLatLongBox):
        """
        Initialize the fuel layer by retrieving the correct fuel data
        by year.

        Arguments:
            LandFireLatLongBox: the retrieved data from Ladfire's python-client

        """
        self.LandFireLatLongBox = LandFireLatLongBox
        self.data = self._get_data()
        self.image = self._make_contours()

    def _get_data(self):
        """
        Functionality to get the raw elevation data for the SimHarness
        """

        # Convert from meters to feet for use with simulator
        data_array = 3.28084 * self.LandFireLatLongBox.topography
        return data_array


class FunctionalTopographyLayer(TopographyLayer):
    """
    Layer that stores elevation data computed from a function.
    """

    def __init__(self, height, width, elevation_fn: ElevationFn, name: str) -> None:
        """
        Initialize the elvation layer by computing the elevations and contours.

        Arguments:
            height: The height of the data layer
            width: The width of the data layer
            elevation_fn: A callable function that converts (x, y) coorindates to
                          elevations.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.name = name

        self.data = self._make_data(elevation_fn)
        self.contours = self._make_contours()

    def _make_data(self, elevation_fn: ElevationFn) -> np.ndarray:
        """
        Use self.elevation_fn to make the elevation data layer.

        Arguments:
            elevation_fn: The function that maps (x, y) points to elevations

        Returns:
            A numpy array containing the elevation data
        """
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)
        elevation_fn_vect = np.vectorize(elevation_fn)
        elevations = elevation_fn_vect(X, Y)
        # Expand third dimension to align with data layers
        elevations = np.expand_dims(elevations, axis=-1)

        return elevations


class FuelLayer(DataLayer):
    """
    Base class for use with operational and procedurally generated
    fuel data. This class implements the code needed to
    create the terrain image to use with the display.
    """

    def __init__(self) -> None:
        """
        Simple call to the parent DataLayer class.

        Arguments:
            None

        Returns:
            None
        """
        super().__init__()
        self.data: np.ndarray
        self.image: np.ndarray

    def _make_image(self) -> np.ndarray:
        """
        Base method to make the terrain background image.

        Arguments:
            None

        Returns:
            A numpy array of the terrain representing an RGB image

        """
        return np.array([])


class OperationalFuelLayer(FuelLayer):
    def __init__(self, LandFireLatLongBox: LandFireLatLongBox):
        """
        Initialize the fuel layer by retrieving the correct fuel data
        by year.

        Arguments:
            LandFireLatLongBox: the retrieved data from Ladfire's python-client

        """
        self.LandFireLatLongBox = LandFireLatLongBox
        self.data = self._get_data()
        self.image = self._make_image()

    def _make_image(self):
        """
        Functionality to convert the raw Fuel Model data
            to rgb values for visualization in the simulator.
        """
        if "fuel" in self.LandFireLatLongBox.layers:
            func = np.vectorize(lambda x: tuple(FuelModelRGB13[x]))
            fuel_data_rgb = func(self.LandFireLatLongBox.fuel)

            fuel_data_rgb = np.stack(fuel_data_rgb, axis=-1) * 255.0

        else:
            fuel_data_rgb = np.array([])

        return fuel_data_rgb

    def _get_data(self):
        """
        Functionality to get the raw Fuel Model data for the Simharness
        """
        func = np.vectorize(lambda x: FuelModelToFuel[x])
        fuel_data = func(self.LandFireLatLongBox.fuel)
        return fuel_data


class FunctionalFuelLayer(FuelLayer):
    """
    Layer that stores fuel data computed from a function.
    """

    def __init__(self, height, width, fuel_fn: FuelArrayFn, name: str) -> None:
        """
        Initialize the fuel layer by computing the fuels.

        Arguments:
            height: The height of the data layer
            width: The width of the data layer
            fuel_fn: A callable function that converts (x, y) coorindates to
                     elevations.
            name: The name of the fuel layer (e.g.: 'chaparral')
        """
        super().__init__()
        self.height = height
        self.width = width
        self.name = name

        self.data = self._make_data(fuel_fn)
        self.texture = self._load_texture()
        self.image = self._make_image()

    def _make_data(self, fuel_fn: FuelArrayFn) -> np.ndarray:
        """
        Use self.fuel_fn to make the fuel data layer.

        Arguments:
            fuel_fn: A callable function that converts (x, y) coorindates to
                     elevations.

        Returns:
            A numpy array containing the fuel data
        """
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)
        fuel_fn_vect = np.vectorize(fuel_fn)
        fuels = fuel_fn_vect(X, Y)
        # Expand third dimension to align with data layers
        fuels = np.expand_dims(fuels, axis=-1)

        return fuels

    def _make_image(self) -> np.ndarray:
        """
        Use the fuel data in self.data to make an RGB background image.

        Returns:
            A NumPy array containing the RGB of the fuel data.
        """
        image = np.zeros((self.width, self.height) + (3,))

        # Loop over the high-level tiles (these are not at the pixel level)
        for i in range(self.height):
            for j in range(self.width):
                # Need these pixel level coordinates to span the correct range
                updated_texture = self._update_texture_dryness(self.data[i][j][0])
                image[i, j] = updated_texture

        return image

    def _update_texture_dryness(self, fuel: Fuel) -> np.ndarray:
        """
        Determine the percent change to make the terrain look drier (i.e.
        more red/yellow/brown) by using the FuelArray values. Then, update
        the texture color using PIL and image blending with a preset
        yellow-brown color/image.

        Arguments:
            fuel: The Fuel with parameters that specify how "dry" the texture should look

        Returns:
            new_texture: The texture with RGB values modified to look drier based
                         on the parameters of fuel_arr
        """
        # Add the numbers after normalization
        # M_x is inverted because a lower value is more flammable
        color_change_pct = fuel.w_0 / 0.2296 + fuel.delta / 7 + (0.2 - fuel.M_x) / 0.2
        # Divide by 3 since there are 3 values
        color_change_pct /= 3

        arr = self.texture.copy()
        arr_img = Image.fromarray(arr)
        resized_brown = DRY_TERRAIN_BROWN_IMG.resize(arr_img.size)
        texture_img = Image.blend(arr_img, resized_brown, color_change_pct / 2)
        new_texture = np.array(texture_img)

        return new_texture

    def _load_texture(self) -> np.ndarray:
        """
        Load the terrain tile texture, resize it to the correct
        shape, and convert to NumPy array

        Returns:
            The returned numpy array of the texture.
        """
        out_size = (1, 1)
        texture = Image.open(TERRAIN_TEXTURE_PATH)
        texture = texture.resize(out_size)
        texture = np.array(texture)

        return texture
