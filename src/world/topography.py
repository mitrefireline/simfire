import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from pathlib import Path
import math

from src.utils.layers import DataLayer


# Developing a function to round to a multiple
def round_up_to_multiple(number, multiple):
    return multiple * math.ceil(number / multiple)


# Developing a function to round to a multiple
def round_down_to_multiple(num, divisor):
    return divisor * math.floor(num / divisor)


class TopoLayer(DataLayer):
    def __init__(self,
                 center: Tuple[float] = (32.1, 115.8),
                 area: int = 1600**2,
                 resolution: int = 30) -> None:
        '''
        This class of methods will get initialized with the config using the lat/long
            bounding box.

        It will get corresponding topographic data from the MERIT DEM:
        5 x 5 degree tiles
        3 arc second (90m) resolution

        It will get corresponding topographic data from the USGS DEM:
        1 x 1 degree tiles
        1/ 3 arc second (10m) resolution

        1 x 1 degree tiles
        1 arc second (30m) resolution

        Arguments:
            BL: A tuple(x1, y1) of latitude/longitude cooordinates

            TR: A tuple(x2, y2) of latitude/longitude cooordinates

            resolution: The type of topography / resolution to get elevation data.
                        Can either be `fine` or `coarse` depending on users needs.

        Return:
            None


        '''
        self.area = area
        self.center = center
        self.resolution = resolution
        self.convert_area()
        self.BR = (self.BL[0], self.TR[1])
        self.TL = (self.TR[0], self.BL[1])
        datapath = Path('/nfs/lslab2/fireline/topographic/')
        try:
            res = str(resolution) + 'm'
            if resolution == 10:
                self.degrees = 1
                self.datapath = datapath / res
                self.width = 10812
                self.height = 10812
            elif resolution == 30:
                self.degrees = 1
                self.datapath = datapath / res
                self.width = 3612
                self.height = 3612
            elif resolution == 90:
                self.degrees = 5
                self.datapath = datapath / res
                self.width = 6000
                self.height = 6000
        except NameError:
            print(f'{resolution} is not available, please selct from: 10m, 30m, 90m.')

    def convert_area(self) -> List[Tuple[float]]:
        '''
        Functionality to use area to create bounding box rather than bottom left
            and top right to specify bounding box

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

        #  bottom_left = (ℎ+12𝐿,𝑘+12𝐿),
        dec_deg = ((1 / 2 * (math.sqrt(self.area))) / self.resolution) * dec_degree_length
        self.BL = (self.center[0] - dec_deg, self.center[1] + dec_deg)
        self.TR = (self.center[0] + dec_deg, self.center[1] - dec_deg)

    def _get_dems(self) -> List[Path]:
        '''
        This method will use the outputed tiles and return the correct dem files

        Arguments:
            None

        Return:
            None

        '''
        self._get_nearest_tile()
        self.tiles = self._stack_tiles()
        self.tif_filenames = []

        for _, ranges in self.tiles.items():
            for range in ranges:
                (five_deg_n, five_deg_w) = range
                tif_data_region = Path(f'n{five_deg_n}w{five_deg_w}.tif')
                tif_file = self.datapath / tif_data_region
                self.tif_filenames.append(tif_file)

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

    def _generate_contours(self) -> np.ndarray:
        '''
        Method to get contour data from corresponding DEM files
            and return as a DataLayer

        Arguments:
            None

        Returns:
            data_array: The data of elevation cnotours for the specified
                        lat / long coordinates
        '''
        self._get_dems()
        elevation_data = Image.open(self.tif_filenames[0])
        elevation_data = np.asarray(elevation_data)
        # flip axis because latitude goes up but numpy will read it down
        elevation_data = np.flip(elevation_data, 0)
        self.data_array = np.expand_dims(elevation_data, axis=-1)

        for key, val in self.tiles.items():
            corners = [(val[0][0], val[0][1]), (val[0][0], val[0][1] - self.degrees),
                       (val[0][0] + self.degrees, val[0][1] - self.degrees),
                       (val[0][0] + self.degrees, val[0][1])]
            if key == 'single':
                # simple case
                self._generate_lat_long(corners)
                tr = (self.bl[0][0], self.tr[1][0])
                bl = (self.tr[0][0], self.bl[1][0])
                return self.data_array[tr[0]:bl[0], tr[1]:bl[1]]
            tmp_array = elevation_data
            for idx, dem in enumerate(self.tif_filenames[1:]):
                elevation_data = Image.open(dem)
                elevation_data = np.asarray(elevation_data)
                # flip axis because latitude goes up but numpy will read it down
                elevation_data = np.flip(elevation_data, 0)
                elevation_data = np.expand_dims(elevation_data, axis=-1)

                if key == 'north':
                    # stack tiles along axis = 0 -> leftmost: bottom, rightmost: top
                    self.data_array = np.concatenate((self.data_array, elevation_data),
                                                     axis=0)
                    corners = self._get_lat_long_bbox(corners, val[idx + 1], key)
                elif key == 'east':
                    # stack tiles along axis = 2 -> leftmost, rightmost
                    self.data_array = np.concatenate((self.data_array, elevation_data),
                                                     axis=1)
                    corners = self._get_lat_long_bbox(corners, val[idx + 1], key)
                elif key == 'square':
                    # stack tiles into a square ->
                    # leftmost: bottom-left, rightmost: top-left
                    corners = self._get_lat_long_bbox(corners, val[idx + 1], key, idx + 1)

                    if idx + 1 == 1:

                        self.data_array = np.concatenate(
                            (self.data_array, elevation_data), axis=1)
                    elif idx + 1 == 2:
                        tmp_array = elevation_data

                    elif idx + 1 == 3:

                        tmp_array = np.concatenate((elevation_data, tmp_array), axis=1)

                        self.data_array = np.concatenate((self.data_array, tmp_array),
                                                         axis=0)
        self._generate_lat_long(corners)
        tr = (self.bl[0][0], self.tr[1][0])
        bl = (self.tr[0][0], self.bl[1][0])
        return self.data_array[tr[0]:bl[0], tr[1]:bl[1]]

        # return self.data_array[self.bl[1][0]:self.tr[1][0], self.bl[0][0]:self.tr[0][0]]
        # return self.data_array[self.tr[0][0]:self.bl[0][0], self.bl[1][0]:self.tr[1][0]]

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
                    self.width = self.width * 3
                else:
                    self.width = self.width * 2
        else:
            self.width = self.width

        if corners[0][0] < corners[2][0]:
            # calculate dimensions (N)
            if corners[2][0] - corners[0][0] > self.degrees:
                if corners[2][0] - corners[0][0] > self.degrees * 3:
                    self.height = self.height * 3
                else:
                    self.height = self.height * 2
        else:
            self.height = self.height

        # create list[Tuple[floats]] with width and height
        y = np.linspace(float(corners[2][0]), float(corners[0][0]), self.height)
        x = np.linspace(float(corners[0][1]), float(corners[1][1]), self.width)
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

        # bl = elev_array_stacked[spatial.KDTree(elev_array_stacked).query(self.BL)[1]]
        # self.bl = np.where((self.elev_array == bl).all(axis=-1))
        # tr = elev_array_stacked[spatial.KDTree(elev_array_stacked).query(self.TR)[1]]
        # self.tr = np.where((self.elev_array == tr).all(axis=-1))

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

    def _save_contour_map(self) -> None:
        '''

        Helper function to generate a contour map of the region
            specified or of the DEM file and save as `<lat_long>.png`

        Elevation in (m)

        Arguments:
            None

        Returns
            None
        '''
        data_array = self._generate_contours()
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

        return data_array
