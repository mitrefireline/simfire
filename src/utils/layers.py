import math
from pathlib import Path
from typing import Tuple, List, Dict

from matplotlib.contour import QuadContourSet
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..enums import (DRY_TERRAIN_BROWN_IMG, TERRAIN_TEXTURE_PATH, FuelModelToFuel)

from ..world.elevation_functions import ElevationFn
from ..world.fuel_array_functions import FuelArrayFn
from ..world.parameters import Fuel


# Developing a function to round to a multiple
def round_up_to_multiple(number, multiple):
    return multiple * math.ceil(number / multiple)


# Developing a function to round to a multiple
def round_down_to_multiple(num, divisor):
    return divisor * math.floor(num / divisor)


class LatLongBox():
    '''
    Class that creates a square coordinate box using a center lat/long point.
    This is used by any DataLayer that needs real coordinates to access data.
    '''
    def __init__(self,
                 center: Tuple[float, float] = (32.1, 115.8),
                 height: int = 1600,
                 width: int = 1600,
                 resolution: int = 30) -> None:
        '''
        This class of methods will get initialized with the config using the lat/long
            bounding box.

        Real-world is measured in meters
        Data is measured in pixels corresponding to the resolution
            i.e: resolution = 10m = 1 pixel

        It will get corresponding topographic data from the MERIT DEM:
        5 x 5 degree tiles
        3 arc second (90m) resolution

        It will get corresponding topographic data from the USGS DEM:
        1 x 1 degree tiles
        1/ 3 arc second (10m) resolution

        1 x 1 degree tiles
        1 arc second (30m) resolution

        Arguments:
            center: The lat/long coordinates of the center point of the screen
            height: The height of one side of the screen (meters)
            width: The width of one side of the screen (meters)
            resolution: The resolution to get data (meters)

        Return:
            None

        TODO: This method only creates a square, needs re-tooling to create a rectangle
        '''
        assert height == width, 'height and width must be equal'

        self.area = height * width
        self.center = center
        self.resolution = resolution

        self.BL, self.TR = self._convert_area()

        self.BR = (self.BL[0], self.TR[1])
        self.TL = (self.TR[0], self.BL[1])
        try:
            if resolution == 10:
                self.degrees = 1
                self.pixel_width = 10812
                self.pixel_height = 10812
            elif resolution == 30:
                self.degrees = 1
                self.pixel_width = 3612
                self.pixel_height = 3612
            elif resolution == 90:
                self.degrees = 5
                self.pixel_width = 6000
                self.pixel_height = 6000
        except NameError:
            print(f'{resolution} is not available, please selct from: 10m, 30m, 90m.')

        self._get_nearest_tile()
        self.tiles = self._stack_tiles()
        self._update_corners()

    def _convert_area(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        '''
        Functionality to use area to create bounding box around the center point
            spanning width x height (meters)

        This function will always make a square.

        Values are found from USGS website for arc-seconds to decimal degrees

        Arguments:
            None
        Return:
            None
        '''
        if self.resolution == 10:
            # convert 5 x 5 degree, 90m resolution into pixel difference
            dec_degree_length = 9.2593e-5
        elif self.resolution == 30:
            # convert 1 x 1 degree, 30m resolution into pixel difference
            dec_degree_length = 0.00027777777803598015
        elif self.resolution == 90:
            # convert 5 x 5 degree, 90m resolution into pixel difference
            dec_degree_length = 0.000833333

        #  bottom_left = (ℎ+12𝐿,𝑘+12𝐿)
        dec_deg = ((1 / 2 * (math.sqrt(self.area))) / self.resolution) * dec_degree_length
        BL = (self.center[0] - dec_deg, self.center[1] + dec_deg)
        TR = (self.center[0] + dec_deg, self.center[1] - dec_deg)

        return BL, TR

    def _get_nearest_tile(self) -> None:
        '''
        This method will take the lat/long tuples and retrieve the nearest dem.
            Always want the lowest (closest to equator and furthest from center divide)
            bound:

        MERIT DEMs are 5 x 5 degrees:
            n30w120 --> N30-N35, W120-W115
            N30-N35, W120-W115 --> (N29.99958333-N34.99958333,W120.0004167-W115.0004167)

        USGUS DEMs are 1 x 1 degrees:
            n34w117 -> (N33 - N34, W116 - W117.00)
            (N33 - N34, W116 - W117.00) -> (N32.999-N34.000, W117.000-W115.999)

        For simplicity, assume we are in upper hemisphere (N) and left of center
            divide (W)

        Arguments:
            None

        Returns:
            None

        '''
        # round up on latitdue
        five_deg_north_min = self.BL[0]
        five_deg_north_min_min = round_up_to_multiple(five_deg_north_min, self.degrees)
        if round(five_deg_north_min_min - five_deg_north_min, 2) <= 0.0001:
            min_max = round_up_to_multiple(five_deg_north_min, self.degrees)
        else:
            min_max = round_down_to_multiple(five_deg_north_min, self.degrees)

        five_deg_north_max = self.TR[0]
        five_deg_north_max_min = round_down_to_multiple(five_deg_north_max, self.degrees)
        if round(five_deg_north_max - five_deg_north_max_min, 2) <= 0.0001:
            max_min = round_up_to_multiple(five_deg_north_max, self.degrees)
        else:
            max_min = round_down_to_multiple(five_deg_north_max, self.degrees)

        self.five_deg_north_min = min_max
        self.five_deg_north_max = max_min

        # round down on longitude (w is negative)
        five_deg_west_max = abs(self.BL[1])
        five_deg_west_max_min = round_down_to_multiple(five_deg_west_max, self.degrees)
        if round(five_deg_west_max - five_deg_west_max_min, 2) <= 0.0001:
            max_min = round_down_to_multiple(five_deg_west_max, self.degrees)
        else:
            max_min = round_up_to_multiple(five_deg_west_max, self.degrees)

        five_deg_west_min = abs(self.TR[1])
        five_deg_west_min_max = round_up_to_multiple(five_deg_west_min, self.degrees)
        if round(five_deg_west_min_max - five_deg_west_min, 2) <= 0.0001:
            min_max = round_down_to_multiple(five_deg_west_min, self.degrees)
        else:
            min_max = round_up_to_multiple(five_deg_west_min, self.degrees)

        self.five_deg_west_min = min_max
        self.five_deg_west_max = max_min

    def _stack_tiles(self) -> Dict[str, Tuple[Tuple[float, float], ...]]:
        '''
        Method to stack DEM tiles correctly. TIles can either be stacked
            starting from bottom left corner:
                Vertically (northernly)
                Horizontally (easternly)
                Square (Mix of easternly and northernly)

            Stacking always follows the standard order:
                bottom left -> bottom right -> top right -> top left

        Arguments:
            None

        Returns:
            A dictionary containing the order to stack the tiles (str)
                and a tuple of tuples of the lat/long (n/w) coordinates of DEM
                tiles (first indice is always bottom left corner).
        '''

        if self.five_deg_north_min == self.five_deg_north_max and \
                self.five_deg_west_max == self.five_deg_west_min:
            # 1 Tile (Simple)
            return {'single': ((self.five_deg_north_min, self.five_deg_west_max), )}

        elif self.five_deg_north_max > self.five_deg_north_min:
            if self.five_deg_north_max - self.five_deg_north_min > self.degrees * 3:
                # 3 Tiles northernly
                return {
                    'north': (
                        (self.five_deg_north_min, self.five_deg_west_max),
                        (self.five_deg_north_max - self.degrees, self.five_deg_west_max),
                        (self.five_deg_north_max, self.five_deg_west_max),
                    )
                }
            elif self.five_deg_west_max > self.five_deg_west_min:
                # 4 Tiles
                return {
                    'square': (
                        (self.five_deg_north_min, self.five_deg_west_max),
                        (self.five_deg_north_min, self.five_deg_west_min),
                        (self.five_deg_north_max, self.five_deg_west_min),
                        (self.five_deg_north_max, self.five_deg_west_max),
                    )
                }
            else:
                # 2 Tiles northernly
                return {
                    'north': (
                        (self.five_deg_north_min, self.five_deg_west_max),
                        (self.five_deg_north_max, self.five_deg_west_max),
                    )
                }
        elif self.five_deg_north_min == self.five_deg_north_max:
            if self.five_deg_west_max > self.five_deg_west_min:
                if self.five_deg_west_max - self.five_deg_west_min > self.degrees:
                    # 3 Tiles easternly
                    return {
                        'east': (
                            (self.five_deg_north_min, self.five_deg_west_max),
                            (self.five_deg_north_min,
                             self.five_deg_west_max - self.degrees),
                            (self.five_deg_north_min, self.five_deg_west_min),
                        )
                    }
                else:
                    # 2 Tiles easternly
                    return {
                        'east': (
                            (self.five_deg_north_min, self.five_deg_west_max),
                            (self.five_deg_north_min, self.five_deg_west_min),
                        )
                    }
            else:
                raise ValueError('The tile stacking failed for parameters '
                                 f'five_deg_north_min: {self.five_deg_north_min}, '
                                 f'five_deg_north_max: {self.five_deg_north_max}, '
                                 f'five_deg_west_min: {self.five_deg_west_min}, '
                                 f'five_deg_west_max: {self.five_deg_west_max}')
        else:
            raise ValueError('The tile stacking failed for parameters '
                             f'five_deg_north_min: {self.five_deg_north_min}, '
                             f'five_deg_north_max: {self.five_deg_north_max}, '
                             f'five_deg_west_min: {self.five_deg_west_min}, '
                             f'five_deg_west_max: {self.five_deg_west_max}')

    def _generate_lat_long(self, corners: List[Tuple[float, float]]) -> None:
        '''
        Use tile name to set bounding box of tile:

        NOTE: We have to manually calculate this because an ArcGIS Pro License is
                required to convert the *.flt Raster files to correct format.

        Resolution: 3-arcseconds = ~0.0008333*3 = 5 deg / 6000 pixels
        Resolution: 1/3-arcseconds = ~0.0008333/3 = 1 deg / 10812 pixels


            n30w120     n30w115

        (35, 120)-----------------(35, 110)
            |------------|------------|
            |-----x------|----x2y2----|
            |------------|------------|
            |------------|------------|
            |------------|------------|
            |----x1y1----|------x-----|
        (30, 120)-----------------(30, 110)

        Arguments:
            corners: A list of the lat/long tuple for each corner in the standard
                     order: [bottom left, bottom right, top right, top left]


        Return:
            None

        '''
        from scipy import spatial

        if corners[0][1] > corners[1][1]:
            # calculate dimensions (W)
            if corners[0][1] - corners[1][1] > self.degrees:
                if corners[0][1] - corners[1][1] > self.degrees * 3:
                    self.pixel_width = self.pixel_width * 3
                else:
                    self.pixel_width = self.pixel_width * 2
        else:
            self.pixel_width = self.pixel_width

        if corners[0][0] < corners[2][0]:
            # calculate dimensions (N)
            if corners[2][0] - corners[0][0] > self.degrees:
                if corners[2][0] - corners[0][0] > self.degrees * 3:
                    self.pixel_height = self.pixel_height * 3
                else:
                    self.pixel_height = self.pixel_height * 2
        else:
            self.pixel_height = self.pixel_height

        # create list[Tuple[floats]] with width and height
        y = np.linspace(float(corners[2][0]), float(corners[0][0]), self.pixel_height)
        x = np.linspace(float(corners[0][1]), float(corners[1][1]), self.pixel_width)
        # rotate to account for (latitude, longitude) -> (y, x)
        XX, YY = np.meshgrid(y, x)
        self.elev_array = np.stack((XX, YY), axis=2)

        # find indices where elevation matches bbox corners
        # this sorts it on the longitude
        elev_array_stacked = np.reshape(
            self.elev_array, (self.elev_array.shape[0] * self.elev_array.shape[1], 2))
        pixels_move = int(np.round((1 / 2 * (math.sqrt(self.area))) / self.resolution))

        center = elev_array_stacked[spatial.KDTree(elev_array_stacked).query(
            self.center)[1]]
        array_center = np.where((self.elev_array == center).all(axis=-1))

        # get tl and br of array indices
        self.tr = (array_center[1] + pixels_move, array_center[0] - pixels_move)
        self.bl = (array_center[1] - pixels_move, array_center[0] + pixels_move)

    def _get_lat_long_bbox(self,
                           corners: List[Tuple[float, float]],
                           new_corner: Tuple[float, float],
                           stack: str,
                           idx: int = 0) -> List[Tuple[float, float]]:
        '''
        This method will update the corners of the array

        Arguments:
            corners: The current corners of the array

            new_corner: A new index to compare the current corners against

            stack: The order in which to stack the tiles and therefore update the
                    corner

            idx: Only used for the 'square' case, to keep track of which tile we are on
                    Tiles are stacked according to standard order:
                    [bottom left, bottom right, top right, top left]

        Returns:
            The indices/bbox of the corners according to standard order:
                [bottom left, bottom right, top right, top left]
        '''
        BL = corners[0]

        BR = corners[1]
        br = (new_corner[0], new_corner[1] - self.degrees)

        TR = corners[2]
        tr = (new_corner[0] + self.degrees, new_corner[1] - self.degrees)

        TL = corners[3]
        tl = (new_corner[0] + self.degrees, new_corner[1])

        if stack == 'east':
            # bottom and top right need to be updated
            return [BL, br, tr, TL]
        elif stack == 'north':
            # top left and right need to be updated
            return [BL, BR, tr, tl]
        elif stack == 'square':
            # where to stack changes at each step
            if idx == 1:
                # Stack the the east
                return [BL, br, tr, TL]
            elif idx == 2:
                # stack to the north
                return [BL, BR, tr, tl]
            else:
                return [BL, BR, TR, tl]
        else:
            raise ValueError('Invalid values for inputs: '
                             f'corners: {corners}, new_corner: {new_corner}, '
                             f'stack: {stack}, idx: {idx}')

    def _update_corners(self) -> None:
        '''
        Method to update corners of total area when 1+ tiles is needed

        Arguments:
            None
        Returns:
            None
        '''

        for key, val in self.tiles.items():
            self.corners = [(val[0][0], val[0][1]), (val[0][0], val[0][1] - self.degrees),
                            (val[0][0] + self.degrees, val[0][1] - self.degrees),
                            (val[0][0] + self.degrees, val[0][1])]
            if key == 'single':
                # simple case
                self._generate_lat_long(self.corners)
            else:
                for idx, _ in enumerate(val[1:]):
                    if key == 'north':
                        # stack tiles along axis = 0 -> leftmost: bottom, rightmost: top
                        self.corners = self._get_lat_long_bbox(self.corners, val[idx + 1],
                                                               key)
                    elif key == 'east':
                        # stack tiles along axis = 2 -> leftmost, rightmost
                        self.corners = self._get_lat_long_bbox(self.corners, val[idx + 1],
                                                               key)
                    elif key == 'square':
                        # stack tiles into a square ->
                        # leftmost: bottom-left, rightmost: top-left
                        self.corners = self._get_lat_long_bbox(self.corners, val[idx + 1],
                                                               key, idx + 1)

        self._generate_lat_long(self.corners)

    def _save_contour_map(self, data_array: np.ndarray, type: str) -> None:
        '''

        Helper function to generate a contour map of the region
            specified or of the DEM file and save as `<lat_long>.png`

        Elevation in (m)

        Arguments:
            None

        Returns
            None
        '''
        import matplotlib.pyplot as plt

        data_array = data_array[:, :, 0]
        # replace missing values if necessary
        if np.any(data_array == -999999.0):
            data_array[data_array == -999999.0] = np.nan

        fig = plt.figure(figsize=(12, 8))
        fig.add_subplot(111)
        if type == 'topo':
            plt.contour(data_array, cmap='viridis')
        else:
            plt.imshow(data_array)
        plt.axis('off')
        plt.title(f'Center: N{self.center[0]}W{self.center[1]}')
        # cbar = plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(f'{type}_n{self.BL[0]}_w{self.BL[1]}_n{self.TR[0]}_w{self.TR[1]}.png')


class DataLayer():
    '''
    Base class for any data that affects the terrain.
    The data in this class should have a value for every pixel in the terrain.
    '''
    def __init__(self) -> None:
        '''
        This parent class only exists to set a base value for self.data
        '''
        self.data: np.ndarray


class TopographyLayer(DataLayer):
    '''
    Base class for use with operational and procedurally generated
    topographic/elevation data. This class implements the code needed to
    create the contour image to use with the display.
    '''
    def __init__(self) -> None:
        '''
        Simple call to the parent DataLayer class.

        Arguments:
            None

        Returns:
            None
        '''
        super().__init__()
        self.image: np.ndarray

    def _make_contours(self) -> QuadContourSet:
        '''
        Use the data in self.data to compute the contour lines.

        Arguments:
            None

        Returns:
            contours: The matplotlib contour set used for plotting
        '''
        contours = plt.contour(self.data.squeeze(), origin='upper')
        plt.close()
        return contours


class OperationalTopographyLayer(TopographyLayer):
    def __init__(self, lat_long_box: LatLongBox) -> None:
        '''
        Initialize the elevation layer by retrieving the correct topograpchic data
            and computing the area.

        Arguments:
            center: The lat/long coordinates of the center point of the screen
            height: The height of the screen size
            width: The width of the screen size
            resolution: The resolution to get data

        '''
        super().__init__()
        self.lat_long_box = lat_long_box
        self.path = Path('/nfs/lslab2/fireline/data/topographic/')
        res = str(self.lat_long_box.resolution) + 'm'
        self.datapath = self.path / res

        self.data = self._make_data()
        self.contours = self._make_contours()

    def _make_data(self) -> np.ndarray:
        self._get_dems()
        data = Image.open(self.tif_filenames[0])
        data = np.asarray(data)
        # flip axis because latitude goes up but numpy will read it down
        data = np.flip(data, 0)
        data = np.expand_dims(data, axis=-1)

        for key, _ in self.lat_long_box.tiles.items():

            if key == 'single':
                # simple case
                tr = (self.lat_long_box.bl[0][0], self.lat_long_box.tr[1][0])
                bl = (self.lat_long_box.tr[0][0], self.lat_long_box.bl[1][0])
                return data[tr[0]:bl[0], tr[1]:bl[1]]
            tmp_array = data
            for idx, dem in enumerate(self.tif_filenames[1:]):
                tif_data = Image.open(dem)
                tif_data = np.asarray(tif_data)
                # flip axis because latitude goes up but numpy will read it down
                tif_data = np.flip(tif_data, 0)
                tif_data = np.expand_dims(tif_data, axis=-1)

                if key == 'north':
                    # stack tiles along axis = 0 -> leftmost: bottom, rightmost: top
                    data = np.concatenate((data, tif_data), axis=0)
                elif key == 'east':
                    # stack tiles along axis = 2 -> leftmost, rightmost
                    data = np.concatenate((data, tif_data), axis=1)
                elif key == 'square':
                    if idx + 1 == 1:
                        data = np.concatenate((data, tif_data), axis=1)
                    elif idx + 1 == 2:
                        tmp_array = tif_data
                    elif idx + 1 == 3:
                        tmp_array = np.concatenate((tif_data, tmp_array), axis=1)
                        data = np.concatenate((data, tmp_array), axis=0)

        tr = (self.lat_long_box.bl[0][0], self.lat_long_box.tr[1][0])
        bl = (self.lat_long_box.tr[0][0], self.lat_long_box.bl[1][0])
        data_array = data[tr[0]:bl[0], tr[1]:bl[1]]
        # Convert from meters to feet for use with Rothermel
        data_array = 3.28084 * data_array
        return data_array

    def _get_dems(self) -> None:
        '''
        This method will use the outputed tiles and return the correct dem files

        Arguments:
            None

        Return:
            None

        '''

        self.tif_filenames = []

        for _, ranges in self.lat_long_box.tiles.items():
            for range in ranges:
                (five_deg_n, five_deg_w) = range
                tif_data_region = Path(f'n{five_deg_n}w{five_deg_w}.tif')
                tif_file = self.datapath / tif_data_region
                self.tif_filenames.append(tif_file)


class FunctionalTopographyLayer(TopographyLayer):
    '''
    Layer that stores elevation data computed from a function.
    '''
    def __init__(self, height, width, elevation_fn: ElevationFn) -> None:
        '''
        Initialize the elvation layer by computing the elevations and contours.

        Arguments:
            height: The height of the data layer
            width: The width of the data layer
            elevation_fn: A callable function that converts (x, y) coorindates to
                          elevations.
        '''
        super().__init__()
        self.height = height
        self.width = width
        self.elevation_fn = elevation_fn

        self.data = self._make_data()
        self.contours = self._make_contours()

    def _make_data(self) -> np.ndarray:
        '''
        Use self.elevation_fn to make the elevation data layer.

        Arguments:
            None

        Returns:
            A numpy array containing the elevation data
        '''
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)
        elevation_fn_vect = np.vectorize(self.elevation_fn)
        elevations = elevation_fn_vect(X, Y)
        # Expand third dimension to align with data layers
        elevations = np.expand_dims(elevations, axis=-1)

        return elevations


class FuelLayer(DataLayer):
    '''
    Base class for use with operational and procedurally generated
    fuel data. This class implements the code needed to
    create the terrain image to use with the display.
    '''
    def __init__(self) -> None:
        '''
        Simple call to the parent DataLayer class.

        Arguments:
            None

        Returns:
            None
        '''
        super().__init__()
        self.image: np.ndarray

    def _make_image(self) -> np.ndarray:
        '''
        Base method to make the terrain background image.

        Arguments:
            None

        Returns:
            A numpy array of the terrain representing an RGB image

        '''
        pass


class OperationalFuelLayer(FuelLayer):
    def __init__(self, lat_long_box: LatLongBox, type: str = '13') -> None:
        '''
        Initialize the elevation layer by retrieving the correct topograpchic data
            and computing the area.

        Arguments:
            center: The lat/long coordinates of the center point of the screen
            height: The height of the screen size
            width: The width of the screen size
            resolution: The resolution to get data

            type: The type of data you wnt to load: 'display' or 'simulation'
                    display: rgb data for rothermel
                    simulation: fuel model values for RL Harness/Simulation

        '''
        self.lat_long_box = lat_long_box
        self.type = type
        # Temporary until we get real fuel data
        self.path = Path('/nfs/lslab2/fireline/data/fuel/')
        res = str(self.lat_long_box.resolution) + 'm'

        self.datapath = self.path / res

        self._get_fuel_dems()
        self.int_data = self._make_data(self.fuel_model_filenames)
        self.data = self._make_fuel_data()
        self.image = self._make_data(self.tif_filenames)
        self.image = self.image * 255.
        self.image = self.image.astype(np.uint8)

    def _make_image(self) -> np.ndarray:
        '''
        Use the fuel data in self.data to make an RGB background image.
        '''
        pass

    def _make_data(self, filename: List) -> np.ndarray:

        data = np.load(filename[0])
        data = np.asarray(data)
        # flip axis because latitude goes up but numpy will read it down
        data = np.flip(data, 0)
        data = np.expand_dims(data, axis=-1)

        for key, _ in self.lat_long_box.tiles.items():

            if key == 'single':
                # simple case
                tr = (self.lat_long_box.bl[0][0], self.lat_long_box.tr[1][0])
                bl = (self.lat_long_box.tr[0][0], self.lat_long_box.bl[1][0])
                return data[tr[0]:bl[0], tr[1]:bl[1]]
            tmp_array = data
            for idx, dem in enumerate(filename[1:]):
                tif_data = np.load(dem)
                tif_data = np.asarray(tif_data)
                # flip axis because latitude goes up but numpy will read it down
                tif_data = np.flip(tif_data, 0)
                tif_data = np.expand_dims(tif_data, axis=-1)

                if key == 'north':
                    # stack tiles along axis = 0 -> leftmost: bottom, rightmost: top
                    data = np.concatenate((data, tif_data), axis=0)
                elif key == 'east':
                    # stack tiles along axis = 2 -> leftmost, rightmost
                    data = np.concatenate((data, tif_data), axis=1)
                elif key == 'square':
                    if idx + 1 == 1:
                        data = np.concatenate((data, tif_data), axis=1)
                    elif idx + 1 == 2:
                        tmp_array = tif_data
                    elif idx + 1 == 3:
                        tmp_array = np.concatenate((tif_data, tmp_array), axis=1)
                        data = np.concatenate((data, tmp_array), axis=0)

        tr = (self.lat_long_box.bl[0][0], self.lat_long_box.tr[1][0])
        bl = (self.lat_long_box.tr[0][0], self.lat_long_box.bl[1][0])
        data_array = data[tr[0]:bl[0], tr[1]:bl[1]]
        return data_array

    def _get_fuel_dems(self) -> None:
        '''
        This method will use the outputed tiles and return the correct dem files
            for both the RGB fuel model data and the fuel model data

        Arguments:
            None

        Return:
            None

        '''

        self.tif_filenames = []
        self.fuel_model_filenames = []
        fuel_model = f'LF2020_FBFM{self.type}_200_CONUS'
        fuel_data_fm = f'LC20_F{self.type}_200_projected_no_whitespace.npy'
        fuel_data_rgb = f'LC20_F{self.type}_200_projected_rgb.npy'
        for _, ranges in self.lat_long_box.tiles.items():
            for range in ranges:
                (five_deg_n, five_deg_w) = range

                int_data_region = Path(
                    f'n{five_deg_n}w{five_deg_w}/{fuel_model}/{fuel_data_fm}')

                rgb_data_region = Path(
                    f'n{five_deg_n}w{five_deg_w}/{fuel_model}/{fuel_data_rgb}')

                int_npy_file = self.datapath / int_data_region
                rgb_npy_file = self.datapath / rgb_data_region
                self.tif_filenames.append(rgb_npy_file)
                self.fuel_model_filenames.append(int_npy_file)

    def _make_fuel_data(self) -> np.ndarray:
        '''
        Map Fire Behavior Fuel Model data to the Fuel type that Rothermel expects

        Arguments:
            None

        Returns:
            np.ndarray: Fuel
        '''
        func = np.vectorize(lambda x: FuelModelToFuel[x])
        data_array = func(self.int_data)
        return data_array


class FunctionalFuelLayer(FuelLayer):
    '''
    Layer that stores fuel data computed from a function.
    '''
    def __init__(self, height, width, fuel_fn: FuelArrayFn) -> None:
        '''
        Initialize the fuel layer by computing the fuels.

        Arguments:
            height: The height of the data layer
            width: The width of the data layer
            fuel_fn: A callable function that converts (x, y) coorindates to
                     elevations.
        '''
        super().__init__()
        self.height = height
        self.width = width
        self.fuel_fn = fuel_fn

        self.data = self._make_data()
        self.texture = self._load_texture()
        self.image = self._make_image()

    def _make_data(self) -> np.ndarray:
        '''
        Use self.fuel_fn to make the fuel data layer.

        Arguments:
            None

        Returns:
            A numpy array containing the fuel data
        '''
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)
        fuel_fn_vect = np.vectorize(self.fuel_fn)
        fuels = fuel_fn_vect(X, Y)
        # Expand third dimension to align with data layers
        fuels = np.expand_dims(fuels, axis=-1)

        return fuels

    def _make_image(self) -> np.ndarray:
        '''
        Use the fuel data in self.data to make an RGB background image.
        '''
        image = np.zeros((self.width, self.height) + (3, ))

        # Loop over the high-level tiles (these are not at the pixel level)
        for i in range(self.height):
            for j in range(self.width):
                # Need these pixel level coordinates to span the correct range
                updated_texture = self._update_texture_dryness(self.data[i][j][0])
                image[i, j] = updated_texture

        return image

    def _update_texture_dryness(self, fuel: Fuel) -> np.ndarray:
        '''
        Determine the percent change to make the terrain look drier (i.e.
        more red/yellow/brown) by using the FuelArray values. Then, update
        the texture color using PIL and image blending with a preset
        yellow-brown color/image.

        Arguments:
            fuel: The Fuel with parameters that specify how "dry" the texture should look

        Returns:
            new_texture: The texture with RGB values modified to look drier based
                         on the parameters of fuel_arr
        '''
        # Add the numbers after normalization
        # M_x is inverted because a lower value is more flammable
        color_change_pct = fuel.w_0 / 0.2296 + \
                           fuel.delta / 7 + \
                           (0.2 - fuel.M_x) / 0.2
        # Divide by 3 since there are 3 values
        color_change_pct /= 3

        arr = self.texture.copy()
        arr_img = Image.fromarray(arr)
        resized_brown = DRY_TERRAIN_BROWN_IMG.resize(arr_img.size)
        texture_img = Image.blend(arr_img, resized_brown, color_change_pct / 2)
        new_texture = np.array(texture_img)

        return new_texture

    def _load_texture(self) -> np.ndarray:
        '''
        Load the terrain tile texture, resize it to the correct
        shape, and convert to numpy

        Returns:
            The returned numpy array of the texture.
        '''
        out_size = (1, 1)
        texture = Image.open(TERRAIN_TEXTURE_PATH)
        texture = texture.resize(out_size)
        texture = np.array(texture)

        return texture
