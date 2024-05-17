import datetime
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import geopandas
import landfire
import matplotlib.pyplot as plt
import numpy as np
from geotiff import GeoTiff
from landfire.product.enums import ProductRegion, ProductTheme, ProductVersion
from landfire.product.search import ProductSearch
from matplotlib.contour import QuadContourSet
from PIL import Image
from scipy.ndimage.morphology import binary_dilation

from ..enums import (
    COLORS,
    DRY_TERRAIN_BROWN_IMG,
    TERRAIN_TEXTURE_PATH,
    BurnStatus,
    FuelModelRGB13,
    FuelModelToFuel,
)
from ..utils.log import create_logger
from ..utils.units import meters_to_feet
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

        log.debug(
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
        # expand "Roads/Urban" fuel type so that fires cannot burn through it
        mask = binary_dilation(self.fuel == 91, [[5, 5, 5]])
        self.fuel[mask] = 91

        self.topography = np.array(self.geotiff_data[:, :, 1])

    def create_lat_lon_array(self) -> np.ndarray:
        """
        We will need to be able to map between the geospatial data and
        the numpy arrays that SimFire uses. To do this, we will create a
        secondary np.ndarray of tuples of lat/lon data.

        This will especially get used with the HistoricalLayer

        Arguments:
            None

        Returns:
            np.ndarray of tuples (h, w): (lat, lon)
        """
        column = np.linspace(
            float(self.points[0][0]), float(self.points[1][0]), self.fuel.shape[0]
        )
        row = np.linspace(
            float(self.points[0][1]), float(self.points[1][1]), self.fuel.shape[1]
        )
        XX, YY = np.meshgrid(row, column)
        # needs to be inverted, but maybe doesn't matter for this....
        lat_lon_data = np.stack((YY, XX))
        lat_lon_data = np.rollaxis(lat_lon_data, axis=0, start=3)
        return lat_lon_data


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
        """Functionality to get the raw elevation data for the SimHarness"""

        # Convert from meters to feet for use with simulator
        data_array = meters_to_feet(self.LandFireLatLongBox.topography)
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


class HistoricalLayer:
    def __init__(
        self,
        year: str,
        state: str,
        fire: str,
        path: Union[Path, str],
        height: int,
        width: int,
    ) -> None:
        """
        Functionality to load the apporpriate/available historical data
        given a fire name, the state, and year of the fire.

        NOTE: Currently, this layer supports adding ALL mitigations to the
            firemap at initialization
        TODO: Add mitigations based upon runtime and implementation time

        NOTE: This layer includes functionality to calculate IoU of final perimeter
        given the BurnMD timestamp and simulation duration for validation
        purposes.
        TODO: Move this functionality to a utilities file
        TODO: Include intermediary perimeters and simulation runtime

        Arguments:
            year: The year of the historical fire.
            state: The state of teh historical fire. This is the full name of the state
                and is case-sensitive.
            fire: The individual fire given the year and state. This is case-sensitive.
            path: The path to the BurnMD dataset.
            height: The height of the screen size (meters).
            width: The width of the screen size (meters).
        """
        self.year = year
        self.state = state
        self.fire = fire
        self.path = path
        self.height = height
        self.width = width

        # Format to convert BurnMD timestamp to datetime object
        self.strptime_fmt = "%Y/%m/%d %H:%M:%S.%f"
        # set the path
        self.fire_path = f"{self.state.title()}/{self.year}/fires/{self.fire.title()}"
        # get available geopandas dataframes
        self._get_historical_data()
        # get fire start location
        self.latitude, self.longitude = self._get_fire_init_pos()
        # get the bounds of screen
        self.points = self._get_bounds_of_screen()
        self.lat_lon_box = LandFireLatLongBox(
            self.points, year=self.year, height=self.height, width=self.width
        )
        self.topography = OperationalTopographyLayer(self.lat_lon_box)
        self.fuel = OperationalFuelLayer(self.lat_lon_box)
        self.lat_lon_array = self.lat_lon_box.create_lat_lon_array()
        self.fire_start_y, self.fire_start_x = get_closest_indice(
            self.lat_lon_array, (self.latitude, self.longitude)
        )
        self.screen_size = self.lat_lon_array.shape[:2]
        self.start_time = self.polygons_df.iloc[0]["DateStart"]
        self.end_time = self.polygons_df.iloc[0]["DateContai"]
        self.perimeter_deltas = self._get_perimeter_time_deltas()
        # get the duraton of fire specified
        self.duration = self._calc_time_elapsed(self.start_time, self.end_time)

    def _get_historical_data(self) -> None:
        """Collect geopandas dataframes availbale for the specified fire"""
        # get the path
        data_path = os.path.join(self.path, self.fire_path)
        try:
            polygons_df = geopandas.read_file(
                os.path.join(data_path, f"{self.fire.title()}_POLYGONS.shp")
            )
        except ValueError:
            polygons_df = geopandas.GeoDataFrame()
            log.warning("There is no perimeter data for this wildfire.")

        try:
            lines_df = geopandas.read_file(
                os.path.join(data_path, f"{self.fire.title()}_LINES.shp")
            )
        except ValueError:
            lines_df = geopandas.GeoDataFrame()
            log.warning("There is no mitigation data for this wildfire.")

        # Add a column with a datetime object for ease of future computation
        polygons_df.insert(
            2,
            "DateTime",
            list(map(self.convert_to_datetime, polygons_df.CreateDate.to_list())),
            True,
        )
        lines_df.insert(
            2,
            "DateTime",
            list(map(self.convert_to_datetime, lines_df.CreateDate.to_list())),
            True,
        )
        self.polygons_df = polygons_df
        self.lines_df = lines_df

    def _get_bounds_of_screen(self):
        """
        Collect the extent of the historical data to set screen

        Returns:
            ((maxy, maxx), (miny, minx))
        """
        if len(self.lines_df) > 0:
            df = self.lines_df
        else:
            df = self.polygons_df

        # most left (west) bound
        maxx = min(
            min(df.geometry.bounds.minx),
            min(df.geometry.bounds.maxx),
        )
        # most right (east) bound
        minx = max(
            max(df.geometry.bounds.maxx),
            max(df.geometry.bounds.minx),
        )
        # most bottom (south) bound
        miny = min(df.geometry.bounds.miny)
        # most top (north) bound
        maxy = max(df.geometry.bounds.maxy)

        return ((maxy, maxx), (miny, minx))

    def _get_fire_init_pos(self):
        """
        Get the embedded fire initial position (approximation)

        Returns:
            (latitude, longitude)
        """
        fire_init_pos = self.polygons_df.iloc[0]["FireInitPo"]
        longitude = float(fire_init_pos.split(", ")[0])
        latitude = float(fire_init_pos.split(", ")[1])

        return latitude, longitude

    def make_mitigations(
        self, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> np.ndarray:
        """
        Method to add mitigation locations to the firemap at the start of the
        simulation.

        Return an array of the mitigation type as an enum that gets passed into the
        sprites.

        Arguments:
            start_time: The start time to grab mitigations for
            end_time: The end time to grab mitigations for

        NOTE: This will not use the time-series aspect of the mitigations

        TODO: Re-write to check if 'Completed Hand' or 'Completed Dozer' lines even exist
        TODO: This will overwrite dozer/hand lines - dozer lines are "stronger" so we
        may want to add logic to keep dozer lines

        Returns:
            An array of mitigations
        """
        mitigation_array = np.zeros((self.screen_size)).astype(int)

        # only going to use `Completed` mitigations
        # we do not care about 'MitigationTimestamps'
        dozer_idxs = np.logical_and.reduce(
            (
                (self.lines_df["FeatureCat"] == "Completed Dozer Line").to_numpy(),
                (self.lines_df["DateTime"] > start_time).to_numpy(),
                (self.lines_df["DateTime"] < end_time).to_numpy(),
            )
        )
        hand_idxs = np.logical_and.reduce(
            (
                (self.lines_df["FeatureCat"] == "Completed Hand Line").to_numpy(),
                (self.lines_df["DateTime"] > start_time).to_numpy(),
                (self.lines_df["DateTime"] < end_time).to_numpy(),
            )
        )
        geo_completed_dozer_mitigations = self.lines_df.loc[dozer_idxs, :]
        geo_completed_hand_mitigations = self.lines_df.loc[hand_idxs, :]

        def _get_geometry(df):
            xy = df.geometry.xy
            longs = xy[0].tolist()
            lats = xy[1].tolist()
            return [list(z) for z in zip(lats, longs)]

        # for mitigation in range(len(geo_completed_dozer_mitigations)):
        dozer_lines = geo_completed_dozer_mitigations.apply(_get_geometry, axis=1)
        hand_lines = geo_completed_hand_mitigations.apply(_get_geometry, axis=1)

        for i in range(len(hand_lines)):
            array_points = []
            points = hand_lines.iloc[i]
            for p in points:
                y, x = get_closest_indice(self.lat_lon_array, (p[1], p[0]))
                array_points.append((y, x))
                mitigation_array[y, x] = BurnStatus.SCRATCHLINE
            # need to interpolate points in this line
            for idx in range(len(array_points) - 1):
                coords = np.linspace(array_points[idx], array_points[idx + 1])
                coords = np.unique(coords.astype(int), axis=0)
                for y, x in coords:
                    mitigation_array[y, x] = BurnStatus.SCRATCHLINE

        for i in range(len(dozer_lines)):
            array_points = []
            points = dozer_lines.iloc[i]
            for p in points:
                y, x = get_closest_indice(self.lat_lon_array, (p[1], p[0]))
                array_points.append((y, x))
                mitigation_array[y, x] = BurnStatus.FIRELINE
            # need to interpolate points in this line
            for idx in range(len(array_points) - 1):
                coords = np.linspace(array_points[idx], array_points[idx + 1])
                coords = np.unique(coords.astype(int), axis=0)
                for y, x in coords:
                    mitigation_array[y, x] = BurnStatus.FIRELINE

        return mitigation_array

    def _calc_time_elapsed(self, start_time: str, end_time: str) -> str:
        """
        Calculate the time between each timestamp with format:
        YYYY/MM/DD HRS:MIN:SEC.0000

        Arguments:
            start_time: beginning of fire as described in BurnMD ddataset
            end_time: end of fire as described by BurnMD dataset

        Returns
            A string of `<>h <>m <>s`
        """
        time_dif = self.convert_to_datetime(end_time) - self.convert_to_datetime(
            start_time
        )

        # convert to days, hours, minutes, seconds
        days = f"{time_dif.days}d"
        datetime_seconds = str(datetime.timedelta(seconds=time_dif.seconds)).split(":")
        hours = f"{datetime_seconds[0]}h"
        minutes = f"{datetime_seconds[1]}m"
        seconds = f"{datetime_seconds[2]}s"

        return days + " " + hours + " " + minutes + " " + seconds

    def convert_to_datetime(self, bmd_time: str) -> datetime.datetime:
        return datetime.datetime.strptime(bmd_time, self.strptime_fmt)

    def _make_perimeters_image(self) -> np.ndarray:
        """
        Create an array of the historical perimeter data

        Returns:
            an array of the historical perimeters
        """
        perimeter_array = np.zeros((self.screen_size)).astype(int)
        geo_perimeters = self.polygons_df.loc[
            self.polygons_df["FeatureCat"] == "Wildfire Daily Fire Perimeter", :
        ]

        def _get_geometry(df):
            xy = df.geometry.exterior.xy
            longs = xy[0].tolist()
            lats = xy[1].tolist()
            return [list(z) for z in zip(lats, longs)]

        perimeters = geo_perimeters.apply(_get_geometry, axis=1)

        for i in range(len(perimeters)):
            array_points = []
            points = perimeters.iloc[i]
            for p in points:
                y, x = get_closest_indice(self.lat_lon_array, (p[1], p[0]))
                array_points.append((x, y))
                # set the value to the perimeter index
                perimeter_array[x, y] = i + 1
            # need to interpolate points in this polygon for connectedness
            for perim_idx in range(len(array_points) - 1):
                coords = np.linspace(
                    array_points[perim_idx], array_points[perim_idx + 1], dtype=int
                )
                coords = np.unique(coords.astype(int), axis=0)
                for coord_x, coord_y in coords:
                    perimeter_array[coord_x, coord_y] = i + 1
        out_image = np.zeros((*perimeter_array.shape, 4), dtype=np.uint8)
        # Map the colors to each index in the fireline points
        np.take(COLORS, perimeter_array, axis=0, out=out_image)

        return out_image

    def _get_perimeter_time_deltas(self):
        """
        Use `_calc_time_elapsed` functionality to get a list of time elapsed between
        perimeters. This can be used in `simulation.run()`to incremently add time to the
        simulation.

        Returns:
            List of time deltas between perimeters
        """

        geo_perimeters = self.polygons_df.loc[
            self.polygons_df["FeatureCat"] == "Wildfire Daily Fire Perimeter", :
        ]

        perimeter_time_deltas = []
        for i in range(len(geo_perimeters)):
            # if first pass, use fire start time to get time delta

            if i == 0:
                delta = self._calc_time_elapsed(
                    geo_perimeters.iloc[i]["DateStart"],
                    geo_perimeters.iloc[i]["PolygonDat"],
                )
            else:
                # TODO do a check to skip an entries that do not have 'PolygonDat' entries
                delta = self._calc_time_elapsed(
                    geo_perimeters.iloc[i - 1]["PolygonDat"],
                    geo_perimeters.iloc[i]["PolygonDat"],
                )

            perimeter_time_deltas.append(delta)
        return perimeter_time_deltas


def get_closest_indice(
    lat_lon_data: np.ndarray, point: Tuple[float, float]
) -> Tuple[int, int]:
    """
    Utility function to help find the closest index for the geospatial point.

    Arguments:
        lat_lon_data: array of the (h, w, (lat, lon)) data of the screen_size of the
            simulation
        point: a tuple pair of lat/lon point. [latitude, longitude]

    Returns:
        y, x: tuple pair of index in lat/lon array that corresponds to
            the simulation array index
    """

    idx = np.argmin(
        np.sqrt(
            np.square(lat_lon_data[..., 0] - point[0])
            + np.square(lat_lon_data[..., 1] - point[1])
        )
    )
    x, y = np.unravel_index(idx, lat_lon_data.shape[:2])

    return int(y), int(x)
