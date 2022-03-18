import numpy as np
from typing import Tuple, List, Dict
from pathlib import Path
import math
from PIL import Image

from ..world.elevation_functions import ElevationFn


# Developing a function to round to a multiple
def round_up_to_multiple(number, multiple):
    return multiple * math.ceil(number / multiple)


# Developing a function to round to a multiple
def round_down_to_multiple(num, divisor):
    return divisor * math.floor(num / divisor)


class DataLayer():
    '''
    Layer class that allows for simulation data layers to be stored and used. A layer
    represents data at every pixel/point in the simulation area.
    '''
    def __init__(self,
                 center: Tuple[float] = (32.1, 115.8),
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
        self.area = height * width
        self.center = center
        self.resolution = resolution

        self.convert_area()

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

    def convert_area(self) -> List[Tuple[float]]:
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
        self.BL = (self.center[0] - dec_deg, self.center[1] + dec_deg)
        self.TR = (self.center[0] + dec_deg, self.center[1] - dec_deg)

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

    def _stack_tiles(self) -> Dict[str, Tuple[Tuple[int]]]:
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

    def _generate_lat_long(self, corners: List[Tuple[int]]) -> Tuple[int]:
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
                           corners: List[Tuple[int]],
                           new_corner: Tuple[int],
                           stack: str,
                           idx: int = 0) -> List[Tuple[int]]:
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

    def _update_corners(self) -> np.ndarray:
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
                for idx in range(len(val)):
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

    def _save_contour_map(self, data_array) -> None:
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
        plt.contour(data_array, cmap='viridis')
        plt.axis('off')
        plt.title(f'Center: N{self.center[0]}W{self.center[1]}')
        # cbar = plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(f'img_n{self.BL[0]}_w{self.BL[1]}_n{self.TR[0]}_w{self.TR[1]}.png')


class TopographyLayer(DataLayer):
    def __init__(self, center: Tuple[float], height: int, width: int,
                 resolution: int) -> None:
        '''
        Initialize the elevation layer by retrieving the correct topograpchic data
            and computing the area.

        Arguments:
            center: The lat/long coordinates of the center point of the screen
            height: The height of the screen size
            width: The width of the screen size
            resolution: The resolution to get data

        '''
        self.path = Path('/nfs/lslab2/fireline/data/topographic/')
        res = str(resolution) + 'm'
        self.datapath = self.path / res
        super().__init__(center, height, width, resolution)
        self.data = self._make_contour_and_data()

    def _make_contour_and_data(self) -> np.ndarray:
        self._get_dems()
        data = Image.open(self.tif_filenames[0])
        data = np.asarray(data)
        # flip axis because latitude goes up but numpy will read it down
        data = np.flip(data, 0)
        data = np.expand_dims(data, axis=-1)

        for key, _ in self.tiles.items():

            if key == 'single':
                # simple case
                tr = (self.bl[0][0], self.tr[1][0])
                bl = (self.tr[0][0], self.bl[1][0])
                return self.data[tr[0]:bl[0], tr[1]:bl[1]]
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
                        tmp_array = data
                    elif idx + 1 == 3:
                        tmp_array = np.concatenate((tif_data, tmp_array), axis=1)
                        self.data = np.concatenate((self.data, tmp_array), axis=0)

        tr = (self.bl[0][0], self.tr[1][0])
        bl = (self.tr[0][0], self.bl[1][0])
        data_array = data[tr[0]:bl[0], tr[1]:bl[1]]
        return data_array

    def _get_dems(self) -> List[Path]:
        '''
        This method will use the outputed tiles and return the correct dem files

        Arguments:
            None

        Return:
            None

        '''

        self.tif_filenames = []

        for _, ranges in self.tiles.items():
            for range in ranges:
                (five_deg_n, five_deg_w) = range
                tif_data_region = Path(f'n{five_deg_n}w{five_deg_w}.tif')
                tif_file = self.datapath / tif_data_region
                self.tif_filenames.append(tif_file)


class FuelLayer(DataLayer):
    def __init__(self, center: Tuple[float], height: int, width: int,
                 resolution: int) -> None:
        '''
        Initialize the elevation layer by retrieving the correct topograpchic data
            and computing the area.

        Arguments:
            center: The lat/long coordinates of the center point of the screen
            height: The height of the screen size
            width: The width of the screen size
            resolution: The resolution to get data

        '''
        self.path = Path('/nfs/lslab2/fireline/fuel/')

        super().__init__(center, height, width, resolution)

    def _make_contour_and_data(self) -> np.ndarray:

        data = Image.open(self.tif_filenames[0])
        data = np.asarray(data)
        # flip axis because latitude goes up but numpy will read it down
        data = np.flip(data, 0)
        self.data = np.expand_dims(data, axis=-1)

        for key, _ in self.tiles.items():

            if key == 'single':
                # simple case
                tr = (self.bl[0][0], self.tr[1][0])
                bl = (self.tr[0][0], self.bl[1][0])
                return self.data[tr[0]:bl[0], tr[1]:bl[1]]
            tmp_array = data
            for idx, dem in enumerate(self.tif_filenames[1:]):
                data = Image.open(dem)
                data = np.asarray(data)
                # flip axis because latitude goes up but numpy will read it down
                data = np.flip(data, 0)
                data = np.expand_dims(data, axis=-1)

                if key == 'north':
                    # stack tiles along axis = 0 -> leftmost: bottom, rightmost: top
                    self.data = np.concatenate((self.data, data), axis=0)
                elif key == 'east':
                    # stack tiles along axis = 2 -> leftmost, rightmost
                    self.data = np.concatenate((self.data, data), axis=1)
                elif key == 'square':
                    if idx + 1 == 1:
                        self.data = np.concatenate((self.data, data), axis=1)
                    elif idx + 1 == 2:
                        tmp_array = data
                    elif idx + 1 == 3:
                        tmp_array = np.concatenate((data, tmp_array), axis=1)
                        self.data = np.concatenate((self.data, tmp_array), axis=0)

        tr = (self.bl[0][0], self.tr[1][0])
        bl = (self.tr[0][0], self.bl[1][0])
        self.data_array = self.data[tr[0]:bl[0], tr[1]:bl[1]]
        return self.data_array


class TransportationLayer(DataLayer):
    def __init__(self, center: Tuple[float], height: int, width: int,
                 resolution: int) -> None:
        '''
        Initialize the elevation layer by retrieving the correct topograpchic data
            and computing the area.

        Arguments:
            center: The lat/long coordinates of the center point of the screen
            height: The height of the screen size
            width: The width of the screen size
            resolution: The resolution to get data

        '''
        self.path = Path('/nfs/lslab2/fireline/transportation/')

        super().__init__(center, height, width, resolution)

    def _make_contour_and_data(self) -> np.ndarray:

        data = Image.open(self.tif_filenames[0])
        data = np.asarray(data)
        # flip axis because latitude goes up but numpy will read it down
        data = np.flip(data, 0)
        self.data = np.expand_dims(data, axis=-1)

        for key, _ in self.tiles.items():

            if key == 'single':
                # simple case
                tr = (self.bl[0][0], self.tr[1][0])
                bl = (self.tr[0][0], self.bl[1][0])
                return self.data[tr[0]:bl[0], tr[1]:bl[1]]
            tmp_array = data
            for idx, dem in enumerate(self.tif_filenames[1:]):
                data = Image.open(dem)
                data = np.asarray(data)
                # flip axis because latitude goes up but numpy will read it down
                data = np.flip(data, 0)
                data = np.expand_dims(data, axis=-1)

                if key == 'north':
                    # stack tiles along axis = 0 -> leftmost: bottom, rightmost: top
                    self.data = np.concatenate((self.data, data), axis=0)
                elif key == 'east':
                    # stack tiles along axis = 2 -> leftmost, rightmost
                    self.data = np.concatenate((self.data, data), axis=1)
                elif key == 'square':
                    if idx + 1 == 1:
                        self.data = np.concatenate((self.data, data), axis=1)
                    elif idx + 1 == 2:
                        tmp_array = data
                    elif idx + 1 == 3:
                        tmp_array = np.concatenate((data, tmp_array), axis=1)
                        self.data = np.concatenate((self.data, tmp_array), axis=0)

        tr = (self.bl[0][0], self.tr[1][0])
        bl = (self.tr[0][0], self.bl[1][0])
        self.data_array = self.data[tr[0]:bl[0], tr[1]:bl[1]]
        return self.data_array


class ElevationLayer(DataLayer):
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
        super().__init__(height, width)
        self.elevation_fn = elevation_fn
        self.data = self._make_contour_and_data()

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

        return elevations
