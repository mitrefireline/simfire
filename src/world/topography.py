import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from pathlib import Path
import math

from utils.layers import DataLayer


# Developing a function to round to a multiple
def round_up_to_multiple(number, multiple):
    return multiple * math.ceil(number / multiple)


# Developing a function to round to a multiple
def round_down_to_multiple(num, divisor):
    return divisor * math.floor(num / divisor)


class MERITLayer(DataLayer):
    def __init__(self, BL: Tuple[float], TR: Tuple[float]) -> None:
        '''

        This class of methods will get initialized with the config.
        It will ingest lat/long cordinates and get corresponding topographic data
            from the MERIT DEM (3 arc second (90m) resolution)

        Arguments:
            x: Tuple(float)
                A tuple(x1, y1) of latitude/longitude cooordinates

            y: Tuple(float)
                A tuple(x2, y2) of latitude/longitude cooordinates

        Return:
            topograpghy: np.ndarray
                The corresponding topography that approximates given lat/long
                    coordinates


        '''

        self.BL = BL
        self.TR = TR
        self.BR = (self.BL[0], self.TR[1])
        self.TL = (self.TR[0], self.BL[1])
        self.datapath = Path('/nfs/lslab2/fireline/topographic/coarse')
        self._get_dems()

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
                tif_data_region = Path(f'n{five_deg_n}w{five_deg_w}_dem.tif')
                tif_file = self.datapath / tif_data_region
                self.tif_filenames.append(tif_file)

    def _get_nearest_tile(self) -> None:
        '''
        This method will take the lat/long tuples and retrieve the nearest dem.

        MERIT DEMs are 5 x 5 degrees:

        Always want the lowest (closest to equator and furthest from center divide) bound:
            n30w120 --> N30-N35, W120-W115
            N30-N35, W120-W115 --> (N29.99958333-N34.99958333,W120.0004167-W115.0004167)

        For simplicity, assume we are in upper hemisphere (N)
            and left of center divide (W)

        Arguments:
            None

        Returns:
            None

        '''
        # round up on latitdue
        five_deg_north_min = self.BL[0]
        five_deg_north_min_min = round_up_to_multiple(five_deg_north_min, 5)
        if round(five_deg_north_min_min - five_deg_north_min, 2) <= 0.01:
            min_max = round_up_to_multiple(five_deg_north_min, 5)
        else:
            min_max = round_down_to_multiple(five_deg_north_min, 5)

        five_deg_north_max = self.TR[0]
        five_deg_north_max_min = round_down_to_multiple(five_deg_north_max, 5)
        if round(five_deg_north_max - five_deg_north_max_min, 2) <= 0.01:
            max_min = round_up_to_multiple(five_deg_north_max, 5)
        else:
            max_min = round_down_to_multiple(five_deg_north_max, 5)

        self.five_deg_north_min = min_max
        self.five_deg_north_max = max_min

        # round down on longitude (w is negative)
        five_deg_west_max = abs(self.BL[1])
        five_deg_west_max_min = round_down_to_multiple(five_deg_west_max, 5)
        if round(five_deg_west_max - five_deg_west_max_min, 2) <= 0.01:
            max_min = round_down_to_multiple(five_deg_west_max, 5)
        else:
            max_min = round_up_to_multiple(five_deg_west_max, 5)

        five_deg_west_min = abs(self.TR[1])
        five_deg_west_min_max = round_up_to_multiple(five_deg_west_min, 5)
        if round(five_deg_west_min_max - five_deg_west_min, 2) <= 0.01:
            min_max = round_down_to_multiple(five_deg_west_min, 5)
        else:
            min_max = round_up_to_multiple(five_deg_west_min, 5)

        self.five_deg_west_min = min_max
        self.five_deg_west_max = max_min

    def _stack_tiles(self) -> Dict[str, Tuple[Tuple[int]]]:
        '''
        Method to stack DEM tiles correctly.

        Logic:

        1 Tile: bbox lies within a single DEM
        2 Tiles: bbox lies either to the east or south of tile
        3 Tiles: bbox lies across 3 tiles to the east or south
        4 Tiles: bbox corners lie across 4 tiles to the south and west

        '''

        if self.five_deg_north_min == self.five_deg_north_max and \
                self.five_deg_west_max == self.five_deg_west_min:
            # 1 Tile (Simple)
            return {'single': ((self.five_deg_north_min, self.five_deg_west_max), )}

        elif self.five_deg_north_max > self.five_deg_north_min:
            if self.five_deg_north_max - self.five_deg_north_min > 5:
                # 3 Tiles northernly
                return {
                    'north': (
                        (self.five_deg_north_min, self.five_deg_west_max),
                        (self.five_deg_north_max - 5, self.five_deg_west_max),
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
                if self.five_deg_west_max - self.five_deg_west_min > 5:
                    # 3 Tiles easternly
                    return {
                        'east': (
                            (self.five_deg_north_min, self.five_deg_west_max),
                            (self.five_deg_north_min, self.five_deg_west_max - 5),
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
            data_array: np.ndarray
                The data of elevation cnotours for the specified lat / long coordinates
        '''
        elevation_data = Image.open(self.tif_filenames[0])
        elevation_data = np.asarray(elevation_data)
        self.data_array = np.expand_dims(elevation_data, axis=-1)
        for key, val in self.tiles.items():
            corners = [(val[0][0], val[0][1]), (val[0][0], val[0][1] - 5),
                       (val[0][0] + 5, val[0][1] - 5), (val[0][0] + 5, val[0][1])]
            if key == 'single':
                # simple case
                self._generate_lat_long(corners)
                return self.data_array[self.bl:self.br, self.tr:self.tl]
            tmp_array = elevation_data
            for idx, dem in enumerate(self.tif_filenames[1:]):
                elevation_data = Image.open(dem)
                elevation_data = np.asarray(elevation_data)
                elevation_data = np.expand_dims(elevation_data, axis=-1)

                if key == 'north':
                    # stack tiles along axis = 0 -> leftmost: bottom, rightmost: top
                    self.data_array = np.vstack((elevation_data, self.data_array))
                    corners = self._get_lat_long_bbox(corners, val[idx + 1], key)
                elif key == 'east':
                    # stack tiles along axis = 2 -> leftmost, rightmost
                    self.data_array = np.hstack((self.data_array, elevation_data))
                    corners = self._get_lat_long_bbox(corners, val[idx + 1], key)
                elif key == 'square':
                    # stack tiles into a square ->
                    # leftmost: bottom-left, rightmost: top-left
                    corners = self._get_lat_long_bbox(corners, val[idx + 1], key, idx)
                    if idx + 1 == 1:
                        self.data_array = np.hstack((elevation_data, self.data_array))
                    elif idx + 1 == 2:
                        tmp_array = elevation_data
                    elif idx + 1 == 3:
                        tmp_array = np.hstack((elevation_data, tmp_array))

                        self.data_array = np.vstack((tmp_array, self.data_array))

        self._generate_lat_long(corners)
        return self.data_array[self.bl:self.br, self.tr:self.tl]

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
        self._get_dems()
        data_array = self._generate_contours()

        # replace missing values if necessary
        if np.any(data_array == -9999.0):
            data_array[data_array == -9999.0] = np.nan
        fig = plt.figure(figsize=(12, 8))
        fig.add_subplot(111)
        plt.contour(data_array, cmap='viridis')  # levels = list(range(0, 6000, 500))
        plt.title('Elevation Contours')
        # cbar = plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(f'img_n{self.BL[0]}_w{self.BL[1]}_n{self.TR[0]}_w{self.TR[1]}.png')

        return data_array

    def _generate_lat_long(self,
                           corners: List[Tuple[int]],
                           height: int = 6000,
                           width: int = 6000) -> Tuple[int]:
        '''
        Use tile name to set bounding box of tile:

        NOTE: We have to manually calculate this because an ArcGIS Pro License is
                required to convert the *.flt Raster files to correct format.

        Resolution: 3-arcseconds = ~0.0008333 = 5 deg / 6000 pixels


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
            corners: List[Tuple[int]]
                A list of the lat/long tuple for each corner in the standard order:
                    [bottom left, bottom right, top right, top left]


        Return:
            None

        '''

        if corners[0][1] > corners[1][1]:
            # calculate dimensions (W)
            if corners[0][1] - corners[1][1] > 5:
                width = width * 2
            else:
                width = 6000

        if corners[0][0] > corners[2][0]:
            # calculate dimensions (N)
            if corners[0][0] - corners[2][0] > 5:
                height = height * 2
            else:
                height = 6000

        # create list[Tuple[floats]] with width and height
        y = np.linspace(float(corners[0][0]), float(corners[2][0]), height)
        x = np.linspace(float(corners[0][1]), float(corners[1][1]), width)
        elev_array = [[(j, i) for j in y] for i in x]
        self.elev_array = np.asarray(elev_array)

        # find indices where elevation matches bbox corners
        # bl = np.unravel_index((np.abs(elev_array - self.BL)).argmin(), elev_array.shape)
        elev_array_stacked = np.reshape(
            self.elev_array, (self.elev_array.shape[0] * self.elev_array.shape[1], 2))
        bl = min(elev_array_stacked,
                 key=lambda c: (c[0] - self.BL[0])**2 + (c[1] - self.BL[1])**2)

        self.bl = np.where((self.elev_array == bl).all(axis=-1))

        br = min(elev_array_stacked,
                 key=lambda c: (c[0] - self.BR[0])**2 + (c[1] - self.BR[1])**2)

        self.br = np.where((self.elev_array == br).all(axis=-1))

        tr = min(elev_array_stacked,
                 key=lambda c: (c[0] - self.TR[0])**2 + (c[1] - self.TR[1])**2)

        self.tr = np.where((self.elev_array == tr).all(axis=-1))

        tl = min(elev_array_stacked,
                 key=lambda c: (c[0] - self.TL[0])**2 + (c[1] - self.TL[1])**2)

        self.tl = np.where((self.elev_array == tl).all(axis=-1))

    def _get_lat_long_bbox(self,
                           corners: List[Tuple[int]],
                           new_corner: Tuple[int],
                           stack: str,
                           idx: int = 0) -> List[Tuple[int]]:
        '''
        This method will update the corners of the array

        Arguments:
            corners: List[Tuple[int]]
                The current corners of the array

            new_corner: Tuple[int]
                A new index to compare the current corners against

            stack: str
                The order in which to stack the tiles and therefore update the
                    corner

            idx: int (Optional)
                Only used for the 'square' case, to keep track of which tile we are on
                    Tiles are stacked according to standard order:
                    [bottom left, bottom right, top right, top left]

        Returns:
            List[Tuple[int]
                The indices/bbox of the corners according to standard order:
                    [bottom left, bottom right, top right, top left]
        '''
        BL = corners[0]

        BR = corners[1]
        br = (new_corner[0], new_corner[1] - 5)

        TR = corners[2]
        tr = (new_corner[0] + 5, new_corner[1] - 5)

        TL = corners[3]
        tl = (new_corner[0] + 5, new_corner[1])

        if stack == 'east':
            # bottom and top right need to be updated
            return [BL, br, tr, TL]
        elif stack == 'north':
            # top left and right need to be updated
            return [BL, BR, tl, tr]
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
